#include "taichi/backends/vulkan/runtime.h"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <chrono>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/vulkan_simple_memory_pool.h"

#include "taichi/math/arithmetic.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {

namespace vulkan {

#define TI_WITH_VULKAN
#ifdef TI_WITH_VULKAN

namespace {
class StopWatch {
 public:
  StopWatch() : begin_(std::chrono::system_clock::now()) {
  }

  int GetMicros() {
    typedef std::chrono::duration<float> fsec;

    auto now = std::chrono::system_clock::now();

    fsec fs = now - begin_;
    begin_ = now;
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(fs);
    return d.count();
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> begin_;
};

using BufferEnum = TaskAttributes::Buffers;
using InputBuffersMap = std::unordered_map<BufferEnum, VkBufferWithMemory *>;

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           Context *host_ctx,
                           uint64_t *host_result_buffer,
                           VkBufferWithMemory *device_buffer)
      : ctx_attribs_(ctx_attribs),
        host_ctx_(host_ctx),
        host_result_buffer_(host_result_buffer),
        device_buffer_(device_buffer) {
  }

  void host_to_device() {
    if (ctx_attribs_->empty()) {
      return;
    }
    auto mapped = device_buffer_->map_mem();
    char *const device_base = reinterpret_cast<char *>(mapped.data());

#define TO_DEVICE(short_type, type)                         \
  else if (dt->is_primitive(PrimitiveTypeID::short_type)) { \
    auto d = host_ctx_->get_arg<type>(i);                   \
    std::memcpy(device_ptr, &d, sizeof(d));                 \
  }
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      const auto dt = arg.dt;
      char *device_ptr = device_base + arg.offset_in_mem;
      if (arg.is_array) {
        const void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(device_ptr, host_ptr, arg.stride);
      }
      TO_DEVICE(i32, int32)
      TO_DEVICE(u32, uint32)
      TO_DEVICE(f32, float32)
      else {
        TI_ERROR("Vulkan does not support arg type={}", data_type_name(arg.dt));
      }
    }
    char *device_ptr = device_base + ctx_attribs_->extra_args_mem_offset();
    std::memcpy(device_ptr, host_ctx_->extra_args,
                ctx_attribs_->extra_args_bytes());
#undef TO_DEVICE
  }

  void device_to_host() {
    if (ctx_attribs_->empty()) {
      return;
    }
    auto mapped = device_buffer_->map_mem();
    char *const device_base = reinterpret_cast<char *>(mapped.data());
#define TO_HOST(short_type, type)                           \
  else if (dt->is_primitive(PrimitiveTypeID::short_type)) { \
    const type d = *reinterpret_cast<type *>(device_ptr);   \
    host_result_buffer_[i] =                                \
        taichi_union_cast_with_different_sizes<uint64>(d);  \
  }
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      char *device_ptr = device_base + arg.offset_in_mem;
      if (arg.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);
      }
    }
    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = device_base + ret.offset_in_mem;
      const auto dt = ret.dt;

      if (ret.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, ret.stride);
      }
      TO_HOST(i32, int32)
      TO_HOST(u32, uint32)
      TO_HOST(f32, float32)
      else {
        TI_ERROR("Vulkan does not support return value type={}",
                 data_type_name(ret.dt));
      }
    }
#undef TO_HOST
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(
      const KernelContextAttributes *ctx_attribs,
      Context *host_ctx,
      uint64_t *host_result_buffer,
      VkBufferWithMemory *device_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, host_ctx, host_result_buffer, device_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  Context *const host_ctx_;
  uint64_t *const host_result_buffer_;
  VkBufferWithMemory *const device_buffer_;
};

// Info for launching a compiled Taichi kernel, which consists of a series of
// Vulkan pipelines.
class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<GlslToSpirvCompiler::SpirvBinary> spirv_bins;
    const SNodeDescriptorsMap *snode_descriptors{nullptr};

    const VulkanDevice *device{nullptr};
    VkBufferWithMemory *root_buffer{nullptr};
    VkBufferWithMemory *global_tmps_buffer{nullptr};
    LinearVkMemoryPool *vk_mem_pool{nullptr};
  };

  CompiledTaichiKernel(const Params &ti_params)
      : ti_kernel_attribs(*ti_params.ti_kernel_attribs) {
    InputBuffersMap input_buffers = {
        {BufferEnum::Root, ti_params.root_buffer},
        {BufferEnum::GlobalTmps, ti_params.global_tmps_buffer},
    };
    if (!ti_kernel_attribs.ctx_attribs.empty()) {
      ctx_buffer_ = ti_params.vk_mem_pool->alloc_and_bind(
          ti_kernel_attribs.ctx_attribs.total_bytes());
      input_buffers[BufferEnum::Context] = ctx_buffer_.get();
    }

    const auto &task_attribs = ti_kernel_attribs.tasks_attribs;
    const auto &spirv_bins = ti_params.spirv_bins;
    TI_ASSERT(task_attribs.size() == spirv_bins.size());

    VulkanCommandBuilder cmd_builder(ti_params.device);
    for (int i = 0; i < task_attribs.size(); ++i) {
      const auto &attribs = task_attribs[i];
      VulkanPipeline::Params vp_params;
      vp_params.device = ti_params.device;
      for (const auto &bb : task_attribs[i].buffer_binds) {
        vp_params.buffer_bindings.push_back(VulkanPipeline::BufferBinding{
            input_buffers.at(bb.type)->buffer(), (uint32_t)bb.binding});
      }
      vp_params.code = SpirvCodeView(spirv_bins[i]);
      auto vp = std::make_unique<VulkanPipeline>(vp_params);
      const int group_x = attribs.advisory_total_num_threads /
                          attribs.advisory_num_threads_per_group;
      cmd_builder.append(*vp, group_x);
      vk_pipelines_.push_back(std::move(vp));
    }
    command_buffer = cmd_builder.build();
  }

  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  TaichiKernelAttributes ti_kernel_attribs;
  std::unique_ptr<VkBufferWithMemory> ctx_buffer_{nullptr};
  std::vector<std::unique_ptr<VulkanPipeline>> vk_pipelines_;
  // VkCommandBuffers are destroyed when the underlying command pool is
  // destroyed.
  // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers#page_Command-buffer-allocation
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};
};

}  // namespace

class VkRuntime ::Impl {
 public:
  explicit Impl(const Params &params)
      : config_(params.config),
        snode_descriptors_(params.snode_descriptors),
        host_result_buffer_(params.host_result_buffer),
        spv_compiler_([](const std::string &glsl_src,
                         const std::string &shader_name,
                         const std::string &err_msg) {
          TI_ERROR_IF("Failed to compile shader={} err={}\n{}", shader_name,
                      err_msg, glsl_src);
        }) {
    TI_ASSERT(config_ != nullptr);
    TI_ASSERT(snode_descriptors_ != nullptr);
    TI_ASSERT(host_result_buffer_ != nullptr);
    VulkanDevice::Params vd_params;
    vd_params.api_version = VulkanEnvSettings::kApiVersion();
    device_ = std::make_unique<VulkanDevice>(vd_params);
    stream_ = std::make_unique<VulkanStream>(device_.get());

    init_memory_pool(params);
    init_vk_buffers();
  }

  ~Impl() {
    {
      decltype(ti_kernels_) tmp;
      tmp.swap(ti_kernels_);
    }
    global_tmps_buffer_.reset();
    root_buffer_.reset();
    dev_local_memory_pool_.reset();
  }

  KernelHandle register_taichi_kernel(RegisterParams reg_params) {
    CompiledTaichiKernel::Params params;
    params.ti_kernel_attribs = &(reg_params.kernel_attribs);
    params.snode_descriptors = snode_descriptors_;
    params.device = device_.get();
    params.root_buffer = root_buffer_.get();
    params.global_tmps_buffer = global_tmps_buffer_.get();
    params.vk_mem_pool = staging_memory_pool_.get();

    for (int i = 0; i < reg_params.task_glsl_source_codes.size(); ++i) {
      const auto &attribs = reg_params.kernel_attribs.tasks_attribs[i];
      const auto &glsl_src = reg_params.task_glsl_source_codes[i];
      const auto &task_name = attribs.name;
      auto spv_bin = spv_compiler_.compile(glsl_src, task_name).value();
      // If we can reach here, we have succeeded. Otherwise
      // std::optional::value() would have killed us.
      TI_INFO("Successfully compiled GLSL -> SPIR-V for task={}\n{}", task_name,
              glsl_src);
      params.spirv_bins.push_back(std::move(spv_bin));
    }
    KernelHandle res;
    res.id_ = ti_kernels_.size();
    ti_kernels_.push_back(std::make_unique<CompiledTaichiKernel>(params));
    return res;
  }

  void launch_kernel(KernelHandle handle, Context *host_ctx) {
    auto *ti_kernel = ti_kernels_[handle.id_].get();
    auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
        &ti_kernel->ti_kernel_attribs.ctx_attribs, host_ctx,
        host_result_buffer_, ti_kernel->ctx_buffer_.get());
    if (ctx_blitter) {
      TI_ASSERT(ti_kernel->ctx_buffer_ != nullptr);
      ctx_blitter->host_to_device();
    }

    stream_->launch(ti_kernel->command_buffer);
    num_pending_kernels_ += ti_kernel->vk_pipelines_.size();
    if (ctx_blitter) {
      synchronize();
      ctx_blitter->device_to_host();
    }
  }

  void synchronize() {
    if (num_pending_kernels_ == 0) {
      return;
    }

    TI_AUTO_PROF;
    StopWatch sw;
    stream_->synchronize();
    TI_INFO("running {} kernels took {} us", num_pending_kernels_,
            sw.GetMicros());
    num_pending_kernels_ = 0;
  }

 private:
  void init_memory_pool(const Params &params) {
    LinearVkMemoryPool::Params mp_params;
    mp_params.physical_device = device_->physical_device();
    mp_params.device = device_->device();
    /*mp_params.poolSize =
        (params.config->device_memory_GB * 1024 * 1024 * 1024ULL);*/
    mp_params.pool_size = 10 * 1024 * 1024;
    mp_params.required_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    mp_params.compute_queue_family_index =
        device_->queue_family_indices().compute_family.value();

    auto &buf_creation_template = mp_params.buffer_creation_template;
    buf_creation_template.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_creation_template.pNext = nullptr;
    buf_creation_template.flags = 0;
    buf_creation_template.size = 0;
    buf_creation_template.usage =
        (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    buf_creation_template.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_creation_template.queueFamilyIndexCount = 1;
    buf_creation_template.pQueueFamilyIndices = nullptr;

    dev_local_memory_pool_ = LinearVkMemoryPool::try_make(mp_params);
    TI_ASSERT(dev_local_memory_pool_ != nullptr);

    mp_params.required_properties = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    buf_creation_template.usage =
        (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    staging_memory_pool_ = LinearVkMemoryPool::try_make(mp_params);
    TI_ASSERT(staging_memory_pool_ != nullptr);
  }

  void init_vk_buffers() {
    root_buffer_ = dev_local_memory_pool_->alloc_and_bind(1024 * 1024);
    global_tmps_buffer_ = dev_local_memory_pool_->alloc_and_bind(1024 * 1024);
  }

  const CompileConfig *const config_;
  const SNodeDescriptorsMap *const snode_descriptors_;
  uint64_t *const host_result_buffer_;

  std::unique_ptr<VulkanDevice> device_{nullptr};
  std::unique_ptr<VulkanStream> stream_{nullptr};
  GlslToSpirvCompiler spv_compiler_;

  std::unique_ptr<LinearVkMemoryPool> dev_local_memory_pool_;
  std::unique_ptr<VkBufferWithMemory> root_buffer_;
  std::unique_ptr<VkBufferWithMemory> global_tmps_buffer_;
  std::unique_ptr<LinearVkMemoryPool> staging_memory_pool_;

  std::vector<std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;
  int num_pending_kernels_{0};
};

#else

class VkRuntime::Impl {
 public:
  Impl(const Params &) {
    TI_ERROR("Vulkan disabled");
  }

  KernelHandle register_taichi_kernel(RegisterParams) {
    TI_ERROR("Vulkan disabled");
    return KernelHandle();
  }

  void launch_kernel(KernelHandle, Context *) {
    TI_ERROR("Vulkan disabled");
  }

  void synchronize() {
    TI_ERROR("Vulkan disabled");
  }

  VkBufferWithMemory *root_buffer() {
    return nullptr;
  }
  VkBufferWithMemory *global_tmps_buffer() {
    return nullptr;
  }
};

#endif  // TI_WITH_VULKAN

VkRuntime::VkRuntime(const Params &params)
    : impl_(std::make_unique<Impl>(params)) {
}

VkRuntime::~VkRuntime() {
}

VkRuntime::KernelHandle VkRuntime::register_taichi_kernel(
    RegisterParams params) {
  return impl_->register_taichi_kernel(std::move(params));
}

void VkRuntime::launch_kernel(KernelHandle handle, Context *host_ctx) {
  impl_->launch_kernel(handle, host_ctx);
}

void VkRuntime::synchronize() {
  impl_->synchronize();
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
