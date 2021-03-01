#define TI_RUNTIME_HOST
#include "cuda_context.h"

#include <unordered_map>
#include <mutex>

#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/util/environ_config.h"
#include "taichi/system/threading.h"
#include "taichi/backends/cuda/cuda_driver.h"

TLANG_NAMESPACE_BEGIN

CUDAContext::CUDAContext()
    : profiler(nullptr), driver(CUDADriver::get_instance_without_context()) {
  // CUDA initialization
  dev_count = 0;
  driver.init(0);
  driver.device_get_count(&dev_count);
  // Unfortunately, at this point |current_program| is still not initialized.
  // So we have to use an env variable.
  if (get_environ_config("TI_REUSE_CUDA_CONTEXT", /*default_value=*/0)) {
    // When the GPU device is configured as EXCLUSIVE_PROCESS mode, only one
    // CUDA context is allowed to be created per device. In this case, the only
    // possible solution for using the CUDA backend is to retrieve and to reuse
    // the current context of the calling CPU thread. See #2190.
    void *existing_context = nullptr;
    driver.context_get_current(&existing_context);
    if (existing_context) {
      // TODO: CUDevice is an alias of uint, not a pointer.
      void *existing_device = nullptr;
      driver.context_get_device(&existing_device);

      device = existing_device;
      context = existing_context;
    } else {
      TI_WARN(
          "Failed to find any existing CUDA context while "
          "TI_REUSE_CUDA_CONTEXT=1");
    }
  }
  if (context == nullptr) {
    // TODO: Support a passed-in device ID.
    // TODO: Check the returned CUresult.
    driver.device_get(&device, /*ordinal=*/0);
    driver.context_create(&context, /*flags=*/0u, device);
  }

  char name[128];
  driver.device_get_name(name, 128, device);

  TI_TRACE("Using CUDA device [id=0]: {}", name);

  int cc_major, cc_minor;
  driver.device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  driver.device_get_attribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

  TI_TRACE("CUDA Device Compute Capability: {}.{}", cc_major, cc_minor);

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  compute_capability = cc_major * 10 + cc_minor;

  if (compute_capability > 75) {
    // The NVPTX backend of LLVM 10.0.0 does not seem to support
    // compute_capability > 75 yet. See
    // llvm-10.0.0.src/build/lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc
    compute_capability = 75;
  }

  mcpu = fmt::format("sm_{}", compute_capability);

  TI_TRACE("Emitting CUDA code for {}", mcpu);
}

std::size_t CUDAContext::get_total_memory() {
  std::size_t ret, _;
  driver.mem_get_info(&_, &ret);
  return ret;
}

std::size_t CUDAContext::get_free_memory() {
  std::size_t ret, _;
  driver.mem_get_info(&ret, &_);
  return ret;
}

void CUDAContext::launch(void *func,
                         const std::string &task_name,
                         std::vector<void *> arg_pointers,
                         unsigned grid_dim,
                         unsigned block_dim,
                         std::size_t shared_mem_bytes) {
  // It is important to keep a handle since in async mode
  // a constant folding kernel may happen during a kernel launch
  // then profiler->start and profiler->stop mismatch.

  KernelProfilerBase::TaskHandle task_handle;
  // Kernel launch
  if (profiler)
    task_handle = profiler->start_with_handle(task_name);
  auto context_guard = CUDAContext::get_instance().get_guard();

  // TODO: remove usages of get_current_program here.
  // Make sure there are not too many threads for the device.
  // Note that the CUDA random number generator does not allow more than
  // [saturating_grid_dim * max_block_dim] threads.
  TI_ASSERT(grid_dim <= get_current_program().config.saturating_grid_dim);
  TI_ASSERT(block_dim <= get_current_program().config.max_block_dim);

  if (grid_dim > 0) {
    std::lock_guard<std::mutex> _(lock);
    driver.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                         shared_mem_bytes, nullptr, arg_pointers.data(),
                         nullptr);
  }
  if (profiler)
    profiler->stop(task_handle);

  if (get_current_program().config.debug) {
    driver.stream_synchronize(nullptr);
  }
}

CUDAContext::~CUDAContext() {
  // TODO: restore these?
  /*
  CUDADriver::get_instance().cuMemFree(context_buffer);
  for (auto cudaModule: cudaModules)
      CUDADriver::get_instance().cuModuleUnload(cudaModule);
  CUDADriver::get_instance().cuCtxDestroy(context);
  */
}

CUDAContext &CUDAContext::get_instance() {
  static auto context = new CUDAContext();
  return *context;
}

TLANG_NAMESPACE_END
