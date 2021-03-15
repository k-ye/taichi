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

#ifndef IN_TAI_VULKAN

#include "taichi/system/profiler.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

#define BAIL_ON_VK_BAD_RESULT(result, msg)        \
  do {                                            \
    TI_ERROR_IF(((result) != VK_SUCCESS), (msg)); \
  } while (0)

#else
#include <iostream>
#define TI_AUTO_PROF
#define TI_INFO(...)
#define BAIL_ON_VK_BAD_RESULT(result, msg) \
  do {                                     \
    if ((result) != VK_SUCCESS) {          \
      throw std::runtime_error((msg));     \
    }                                      \
  } while (0)

#endif  // IN_TAI_VULKAN

#include "taichi/math/arithmetic.h"

namespace taichi {
namespace lang {
namespace vulkan {

#define TI_WITH_VULKAN
#ifdef TI_WITH_VULKAN

namespace {

constexpr VkAllocationCallbacks *kNoVkAllocCallbacks = nullptr;
constexpr bool kEnableValidationLayers = true;

constexpr std::array<const char *, 1> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

bool CheckValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  std::unordered_set<std::string> availableLayerNames;
  for (const auto &layerProps : availableLayers) {
    availableLayerNames.insert(layerProps.layerName);
  }
  for (const char *name : kValidationLayers) {
    if (availableLayerNames.count(std::string(name)) == 0) {
      return false;
    }
  }
  return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                  void *pUserData) {
  if (messageSeverity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  }
  return VK_FALSE;
}

void PopulateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT *createInfo) {
  *createInfo = {};
  createInfo->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo->pfnUserCallback = vk_debug_callback;
  createInfo->pUserData = nullptr;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

std::vector<const char *> GetRequiredExtensions() {
  std::vector<const char *> extensions;
  if constexpr (kEnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

std::vector<const char *> GetDeviceRequiredExtensions() {
  std::vector<const char *> extensions;
  // extensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
  return extensions;
}

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

struct QueueFamilyIndices {
  std::optional<uint32_t> computeFamily;

  bool IsComplete() const {
    return computeFamily.has_value();
  }
};

QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queue_families.data());

  constexpr VkQueueFlags kFlagMask =
      (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT));

  // first try and find a queue that has just the compute bit set
  for (int i = 0; i < queueFamilyCount; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if ((masked_flags & VK_QUEUE_COMPUTE_BIT) &&
        !(masked_flags & VK_QUEUE_GRAPHICS_BIT)) {
      indices.computeFamily = i;
    }
    if (indices.IsComplete()) {
      return indices;
    }
  }

  // lastly get any queue that will work
  for (int i = 0; i < queueFamilyCount; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if (masked_flags & VK_QUEUE_COMPUTE_BIT) {
      indices.computeFamily = i;
    }
    if (indices.IsComplete()) {
      return indices;
    }
  }
  return indices;
}

bool IsDeviceSuitable(VkPhysicalDevice device) {
  const QueueFamilyIndices indices = FindQueueFamilies(device);
  return indices.IsComplete();
}

VkShaderModule CreateShaderModule(VkDevice device, const SpirvCodeView &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size;
  createInfo.pCode = code.data;

  VkShaderModule shaderModule;
  BAIL_ON_VK_BAD_RESULT(
      vkCreateShaderModule(device, &createInfo, kNoVkAllocCallbacks,
                           &shaderModule),
      "failed to create shader module");
  return shaderModule;
}

uint32_t FindMemoryType(VkPhysicalDevice physicalDevice,
                        uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

/// Taichi Helpers

using BufferEnum = TaskAttributes::Buffers;

struct VkBufferWithSize {
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceSize size{0};
};

using InputBuffersMap = std::unordered_map<BufferEnum, VkBufferWithSize>;

class NaiveVkBufferAllocator {
 public:
  struct Params {
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
  };

  explicit NaiveVkBufferAllocator(const Params &params);
  ~NaiveVkBufferAllocator();
  NaiveVkBufferAllocator(const NaiveVkBufferAllocator &) = delete;
  NaiveVkBufferAllocator &operator=(const NaiveVkBufferAllocator &) = delete;
  NaiveVkBufferAllocator(NaiveVkBufferAllocator &&) = default;
  NaiveVkBufferAllocator &operator=(NaiveVkBufferAllocator &&) = default;

  VkBufferWithSize alloc_and_bind(VkDeviceSize size);

 private:
  VkDevice device_{VK_NULL_HANDLE};  // not owned
  VkBufferCreateInfo bufCreateInfoTemplate_{};
  VkDeviceMemory memoryPool_{VK_NULL_HANDLE};
  VkDeviceSize alignment_{0};
  VkDeviceSize next_{0};
};

NaiveVkBufferAllocator::NaiveVkBufferAllocator(const Params &params)
    : device_(params.device) {
  // Create a dummy buffer
  bufCreateInfoTemplate_ = VkBufferCreateInfo{};
  bufCreateInfoTemplate_.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufCreateInfoTemplate_.pNext = nullptr;
  bufCreateInfoTemplate_.size = 1024;
  bufCreateInfoTemplate_.flags = 0;
  bufCreateInfoTemplate_.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufCreateInfoTemplate_.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  bufCreateInfoTemplate_.queueFamilyIndexCount = 0;
  bufCreateInfoTemplate_.pQueueFamilyIndices = nullptr;

  VkBuffer dummyBuffer{VK_NULL_HANDLE};
  BAIL_ON_VK_BAD_RESULT(vkCreateBuffer(device_, &bufCreateInfoTemplate_,
                                       kNoVkAllocCallbacks, &dummyBuffer),
                        "failed to create buffer");
  // Allocate Vulkan buffer memory
  VkMemoryRequirements memRequirements{};
  vkGetBufferMemoryRequirements(device_, dummyBuffer, &memRequirements);
  alignment_ = memRequirements.alignment;

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = 64 * 1024 * 1024;
  allocInfo.memoryTypeIndex =
      FindMemoryType(params.physicalDevice, memRequirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  BAIL_ON_VK_BAD_RESULT(
      vkAllocateMemory(device_, &allocInfo, kNoVkAllocCallbacks, &memoryPool_),
      "failed to allocate buffer memory");
  vkDestroyBuffer(device_, dummyBuffer, kNoVkAllocCallbacks);
}

NaiveVkBufferAllocator::~NaiveVkBufferAllocator() {
  vkFreeMemory(device_, memoryPool_, kNoVkAllocCallbacks);
}

VkBufferWithSize NaiveVkBufferAllocator::alloc_and_bind(VkDeviceSize size) {
  size = iroundup(size, alignment_);
  bufCreateInfoTemplate_.size = size;
  VkBuffer buffer{VK_NULL_HANDLE};
  BAIL_ON_VK_BAD_RESULT(vkCreateBuffer(device_, &bufCreateInfoTemplate_,
                                       kNoVkAllocCallbacks, &buffer),
                        "failed to create buffer");

  BAIL_ON_VK_BAD_RESULT(vkBindBufferMemory(device_, buffer, memoryPool_, next_),
                        "failed to bind buffer to memory");
  next_ += size;
  return VkBufferWithSize{buffer, size};
}

class UserVulkanKernel {
 public:
  struct Params {
    SpirvCodeView code;
    const TaskAttributes *attribs = nullptr;
    const InputBuffersMap *input_buffers = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
  };

  explicit UserVulkanKernel(const Params &params)
      : name_(params.attribs->name),
        device_(params.device),
        compute_queue_(params.compute_queue) {
    CreateDescriptorSetLayout(params);
    CreateComputePipeline(params);
    CreateDescriptorPool(params);
    CreateDescriptorSets(params);
    CreateCommandBuffer(params);
  }

  ~UserVulkanKernel() {
    // Command buffers will be automatically freed when their command pool is
    // destroyed, so we don't need an explicit cleanup.
    vkDestroyDescriptorPool(device_, descriptorPool_, kNoVkAllocCallbacks);
    vkDestroyPipeline(device_, pipeline_, kNoVkAllocCallbacks);
    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_,
                                 kNoVkAllocCallbacks);
  }

  void launch() {
    TI_AUTO_PROF;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;
    BAIL_ON_VK_BAD_RESULT(vkQueueSubmit(compute_queue_, /*submitCount=*/1,
                                        &submitInfo, /*fence=*/VK_NULL_HANDLE),
                          "failed to submit command buffer");
  }

 private:
  void CreateDescriptorSetLayout(const Params &params) {
    const auto &buffer_binds = params.attribs->buffer_binds;
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.reserve(buffer_binds.size());
    for (const auto &bb : buffer_binds) {
      VkDescriptorSetLayoutBinding layoutBinding{};
      layoutBinding.binding = bb.binding;
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layoutBinding.descriptorCount = 1;
      layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      layoutBinding.pImmutableSamplers = nullptr;
      layoutBindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = layoutBindings.size();
    layoutCreateInfo.pBindings = layoutBindings.data();

    BAIL_ON_VK_BAD_RESULT(
        vkCreateDescriptorSetLayout(device_, &layoutCreateInfo,
                                    kNoVkAllocCallbacks, &descriptorSetLayout_),
        "failed to create descriptor set layout");
  }

  void CreateComputePipeline(const Params &params) {
    VkShaderModule shaderModule = CreateShaderModule(device_, params.code);

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    BAIL_ON_VK_BAD_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutInfo,
                               kNoVkAllocCallbacks, &pipelineLayout_),
        "failed to create pipeline layout");

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout_;
    BAIL_ON_VK_BAD_RESULT(
        vkCreateComputePipelines(device_, /*pipelineCache=*/VK_NULL_HANDLE,
                                 /*createInfoCount=*/1, &pipelineInfo,
                                 kNoVkAllocCallbacks, &pipeline_),
        "failed to create pipeline");

    vkDestroyShaderModule(device_, shaderModule, kNoVkAllocCallbacks);
  }

  void CreateDescriptorPool(const Params &params) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // Total number of descriptors across all the descriptor sets that can be
    // allocated from this pool. See
    // https://www.reddit.com/r/vulkan/comments/8u9zqr/having_trouble_understanding_descriptor_pool/e1e8d5f?utm_source=share&utm_medium=web2x&context=3
    poolSize.descriptorCount = params.attribs->buffer_binds.size();

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    BAIL_ON_VK_BAD_RESULT(
        vkCreateDescriptorPool(device_, &poolInfo, kNoVkAllocCallbacks,
                               &descriptorPool_),
        "failed to create descriptor pool");
  }

  void CreateDescriptorSets(const Params &params) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;
    BAIL_ON_VK_BAD_RESULT(
        vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_),
        "failed to allocate descriptor set");

    const auto &buffer_binds = params.attribs->buffer_binds;
    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
    descriptorBufferInfos.reserve(buffer_binds.size());
    for (const auto &bb : buffer_binds) {
      auto &bm = params.input_buffers->at(bb.type);
      VkDescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = bm.buffer;
      // Note that this is the offset within the buffer itself, not the offset
      // of this buffer within its backing memory!
      bufferInfo.offset = 0;
      bufferInfo.range = bm.size;
      descriptorBufferInfos.push_back(bufferInfo);
    }

    // https://software.intel.com/content/www/us/en/develop/articles/api-without-secrets-introduction-to-vulkan-part-6.html
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    descriptorWrites.reserve(descriptorBufferInfos.size());
    for (int i = 0; i < buffer_binds.size(); ++i) {
      const auto &bb = buffer_binds[i];

      VkWriteDescriptorSet write{};
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.dstSet = descriptorSet_;
      write.dstBinding = bb.binding;
      write.dstArrayElement = 0;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write.pBufferInfo = &descriptorBufferInfos[i];
      write.pImageInfo = nullptr;
      write.pTexelBufferView = nullptr;
      descriptorWrites.push_back(write);
    }

    vkUpdateDescriptorSets(device_,
                           /*descriptorWriteCount=*/descriptorWrites.size(),
                           descriptorWrites.data(), /*descriptorCopyCount=*/0,
                           /*pDescriptorCopies=*/nullptr);
  }

  void CreateCommandBuffer(const Params &params) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = params.command_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    BAIL_ON_VK_BAD_RESULT(
        vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer_),
        "failed to allocate command buffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // This flag allows us to submit the same command buffer to the queue
    // multiple times, while they are still pending.
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = nullptr;
    BAIL_ON_VK_BAD_RESULT(vkBeginCommandBuffer(commandBuffer_, &beginInfo),
                          "failed to begin recording command buffer");

    const auto *attribs = params.attribs;
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
        /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptorSet_,
        /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/nullptr);
    const auto group_x = attribs->advisory_total_num_threads /
                         attribs->advisory_num_threads_per_group;
    TI_INFO(
        "Created command buffer for kernel={} total_num={} local_group_size={} "
        "num_group_x={}",
        attribs->name, attribs->advisory_total_num_threads,
        attribs->advisory_num_threads_per_group, group_x);
    std::cout << "Created command buffer for kernel=" << attribs->name
              << " total_num_threads=" << attribs->advisory_total_num_threads
              << " local_group_size=" << attribs->advisory_num_threads_per_group
              << " num_group_x=" << group_x << std::endl;
    vkCmdDispatch(commandBuffer_, group_x,
                  /*groupCountY=*/1,
                  /*groupCountZ=*/1);

    // #warning "TODO: Add vkCmdPipelineBarrier()"
    // Copied from TVM
    // https://github.com/apache/tvm/blob/b2a3c481ebbb7cfbd5335fb11cd516ae5f348406/src/runtime/vulkan/vulkan.cc#L1134-L1142
    VkMemoryBarrier barrier_info{};
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;
    barrier_info.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier_info.dstAccessMask =
        (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(commandBuffer_,
                         /*srcStageMask=*/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         /*srcStageMask=*/0, /*memoryBarrierCount=*/1,
                         &barrier_info, /*bufferMemoryBarrierCount=*/0,
                         /*pBufferMemoryBarriers=*/nullptr,
                         /*imageMemoryBarrierCount=*/0,
                         /*pImageMemoryBarriers=*/nullptr);
    BAIL_ON_VK_BAD_RESULT(vkEndCommandBuffer(commandBuffer_),
                          "failed to record command buffer");
  }

  std::string name_;
  VkDevice device_;        // not owned
  VkQueue compute_queue_;  // not owned
  VkDescriptorSetLayout descriptorSetLayout_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline pipeline_;
  VkDescriptorPool descriptorPool_;
  VkDescriptorSet descriptorSet_;
  VkCommandBuffer commandBuffer_;
};

#if 0
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
    return;
    TI_AUTO_PROF;
    if (ctx_attribs_->empty()) {
      return;
    }
    StopWatch sw;
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
    TI_INFO("H2D memory copy, duration={} us", sw.GetMicros());
#undef TO_DEVICE
  }

  void device_to_host() {
    return;
    TI_AUTO_PROF;
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
    StopWatch sw;
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
    TI_INFO("D2H memory copy, duration={} us", sw.GetMicros());

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
#endif

// Info for launching a compiled Taichi kernel, which consists of a series of
// compiled Vulkan kernels.
class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs = nullptr;
    std::vector<GlslToSpirvCompiler::SpirvBinary> spirv_bins;
    VkBufferWithSize root_buffer;
    VkBufferWithSize global_tmps_buffer;
    VkBufferWithSize context_buffer;
    NaiveVkBufferAllocator *vk_mem_pool = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
  };

  CompiledTaichiKernel(const Params &ti_params)
      : ti_kernel_attribs(*ti_params.ti_kernel_attribs),
        device_(ti_params.device) {
    InputBuffersMap input_buffers = {
        {BufferEnum::Root, ti_params.root_buffer},
        {BufferEnum::GlobalTmps, ti_params.global_tmps_buffer},
        {BufferEnum::Context, ti_params.context_buffer},
    };
    if (!ti_kernel_attribs.ctx_attribs.empty()) {
      ctx_buffer_ = ti_params.vk_mem_pool->alloc_and_bind(
          ti_kernel_attribs.ctx_attribs.total_bytes());
      input_buffers[BufferEnum::Context] = ctx_buffer_.value();
    }

    const auto &task_attribs = ti_kernel_attribs.tasks_attribs;
    const auto &spirv_bins = ti_params.spirv_bins;
    TI_ASSERT(task_attribs.size() == spirv_bins.size());

    for (int i = 0; i < task_attribs.size(); ++i) {
      UserVulkanKernel::Params vk_params;
      vk_params.code = SpirvCodeView(spirv_bins[i]);
      vk_params.attribs = &task_attribs[i];
      vk_params.input_buffers = &input_buffers;
      vk_params.device = ti_params.device;
      vk_params.compute_queue = ti_params.compute_queue;
      vk_params.command_pool = ti_params.command_pool;

      vk_kernels.push_back(
          std::make_unique<UserVulkanKernel>(std::move(vk_params)));
    }
  }

  ~CompiledTaichiKernel() {
    if (ctx_buffer_.has_value()) {
      vkDestroyBuffer(device_, ctx_buffer_->buffer, kNoVkAllocCallbacks);
    }
  }
  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  TaichiKernelAttributes ti_kernel_attribs;
  std::optional<VkBufferWithSize> ctx_buffer_;
  std::vector<std::unique_ptr<UserVulkanKernel>> vk_kernels;

 private:
  VkDevice device_{VK_NULL_HANDLE};  // Not owned
};

}  // namespace

class VkRuntime ::Impl {
 public:
  explicit Impl(const Params &params)
      : config_(params.config), host_result_buffer_(params.host_result_buffer) {
    // TI_ASSERT(config_ != nullptr);
    // TI_ASSERT(host_result_buffer_ != nullptr);
    CreateInstance();
    SetupDebugMessenger();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateCommandPool();
    init_memory_pool(params);
    init_vk_buffers();
  }

  ~Impl() {
    {
      decltype(ti_kernels_) tmp;
      tmp.swap(ti_kernels_);
    }
    vkDestroyBuffer(device_, fake_ctx_buffer_.buffer, kNoVkAllocCallbacks);
    vkDestroyBuffer(device_, global_tmps_buffer_.buffer, kNoVkAllocCallbacks);
    vkDestroyBuffer(device_, root_buffer_.buffer, kNoVkAllocCallbacks);
    memory_pool_.reset();
    if constexpr (kEnableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_,
                                    kNoVkAllocCallbacks);
    }
    vkDestroyCommandPool(device_, command_pool_, kNoVkAllocCallbacks);
    vkDestroyDevice(device_, kNoVkAllocCallbacks);
    vkDestroyInstance(instance_, kNoVkAllocCallbacks);
  }

  KernelHandle register_taichi_kernel(RegisterParams reg_params) {
    CompiledTaichiKernel::Params params;
    params.ti_kernel_attribs = &(reg_params.kernel_attribs);
    params.root_buffer = root_buffer_;
    params.global_tmps_buffer = global_tmps_buffer_;
    params.context_buffer = fake_ctx_buffer_;
    params.vk_mem_pool = memory_pool_.get();
    params.device = device_;
    params.compute_queue = compute_queue_;
    params.command_pool = command_pool_;

    for (int i = 0; i < reg_params.task_glsl_source_codes.size(); ++i) {
      const auto &attribs = reg_params.kernel_attribs.tasks_attribs[i];
      auto spv_bin =
          spv_compiler_
              .compile(reg_params.task_glsl_source_codes[i], attribs.name)
              .value();
      params.spirv_bins.push_back(std::move(spv_bin));
    }
    KernelHandle res;
    res.id_ = ti_kernels_.size();
    ti_kernels_.push_back(std::make_unique<CompiledTaichiKernel>(params));
    return res;
  }

  void launch_kernel(KernelHandle handle, Context *host_ctx) {
    // ti_kernels_[handle.id_]->launch();
    auto *ti_kernel = ti_kernels_[handle.id_].get();
    // auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
    //     &ti_kernel->ti_kernel_attribs.ctx_attribs, host_ctx,
    //     host_result_buffer_, ti_kernel->ctx_buffer_.get());
    // if (ctx_blitter) {
    //   TI_ASSERT(ti_kernel->ctx_buffer_ != nullptr);
    //   ctx_blitter->host_to_device();
    // }
    int i = 0;
    for (auto &vk : ti_kernel->vk_kernels) {
      vk->launch();
      pending_kernels_.push_back(
          ti_kernel->ti_kernel_attribs.tasks_attribs[i].name);
      ++i;
    }
    // if (ctx_blitter) {
    //   sync(/*for_ctx=*/true);
    //   ctx_blitter->device_to_host();
    // }
  }

  void synchronize() {
    sync(/*for_ctx=*/false);
  }

 private:
  void sync(bool for_ctx) {
    TI_AUTO_PROF;
    TI_ASSERT(!pending_kernels_.empty());
    StopWatch sw;
    vkQueueWaitIdle(compute_queue_);
    TI_INFO("Vulkan: sync {} kernels took {} us", pending_kernels_.size(),
            sw.GetMicros());
    pending_kernels_.clear();
  }

  void CreateInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Taichi Vulkan Backend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VulkanEnvSettings::kApiVersion();  // important

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if constexpr (kEnableValidationLayers) {
      TI_ASSERT_INFO(CheckValidationLayerSupport(),
                     "validation layers requested but not available");
    }
    // for (const auto &ext : GetInstanceExtensionProperties()) {
    //   std::cout << "instancce ext=" << ext.extensionName
    //             << " spec=" << ext.specVersion << std::endl;
    // }
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

    if constexpr (kEnableValidationLayers) {
      createInfo.enabledLayerCount = (uint32_t)kValidationLayers.size();
      createInfo.ppEnabledLayerNames = kValidationLayers.data();

      PopulateDebugMessengerCreateInfo(&debugCreateInfo);
      createInfo.pNext = &debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }
    const auto extensions = GetRequiredExtensions();
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    BAIL_ON_VK_BAD_RESULT(
        vkCreateInstance(&createInfo, kNoVkAllocCallbacks, &instance_),
        "failed to create instance");
  }

  void SetupDebugMessenger() {
    if (!kEnableValidationLayers) {
      return;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    PopulateDebugMessengerCreateInfo(&createInfo);

    BAIL_ON_VK_BAD_RESULT(
        CreateDebugUtilsMessengerEXT(instance_, &createInfo,
                                     kNoVkAllocCallbacks, &debugMessenger_),
        "failed to set up debug messenger");
  }

  void PickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    TI_ASSERT_INFO(deviceCount > 0, "failed to find GPUs with Vulkan support");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());
    physicalDevice_ = VK_NULL_HANDLE;
    for (const auto &device : devices) {
      if (IsDeviceSuitable(device)) {
        physicalDevice_ = device;
        break;
      }
    }
    TI_ASSERT_INFO(physicalDevice_ != VK_NULL_HANDLE,
                   "failed to find a suitable GPU");

    queueFamilyIndices_ = FindQueueFamilies(physicalDevice_);
  }

  void CreateLogicalDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex =
        queueFamilyIndices_.computeFamily.value();
    queueCreateInfo.queueCount = 1;
    constexpr float kQueuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &kQueuePriority;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;

    VkPhysicalDeviceFeatures deviceFeatures{};
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;

    if (kEnableValidationLayers) {
      createInfo.enabledLayerCount = (uint32_t)kValidationLayers.size();
      createInfo.ppEnabledLayerNames = kValidationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }
    BAIL_ON_VK_BAD_RESULT(vkCreateDevice(physicalDevice_, &createInfo,
                                         kNoVkAllocCallbacks, &device_),
                          "failed to create logical device");
    vkGetDeviceQueue(device_, queueFamilyIndices_.computeFamily.value(),
                     /*queueIndex=*/0, &compute_queue_);
  }

  void CreateCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.queueFamilyIndex = queueFamilyIndices_.computeFamily.value();
    BAIL_ON_VK_BAD_RESULT(
        vkCreateCommandPool(device_, &poolInfo, kNoVkAllocCallbacks,
                            &command_pool_),
        "failed to create command pool");
  }

  void init_memory_pool(const Params &) {
    NaiveVkBufferAllocator::Params mparams;
    mparams.physicalDevice = physicalDevice_;
    mparams.device = device_;
    memory_pool_ = std::make_unique<NaiveVkBufferAllocator>(mparams);
  }

  void init_vk_buffers() {
    root_buffer_ = memory_pool_->alloc_and_bind(1024 * 1024);
    global_tmps_buffer_ = memory_pool_->alloc_and_bind(1024 * 1024);
    fake_ctx_buffer_ = memory_pool_->alloc_and_bind(1024 * 1024);
  }

  const CompileConfig *const config_;
  uint64_t *const host_result_buffer_;
  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debugMessenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
  QueueFamilyIndices queueFamilyIndices_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkCommandPool command_pool_{VK_NULL_HANDLE};
  GlslToSpirvCompiler spv_compiler_;

  // std::unique_ptr<LinearVkMemoryPool> memory_pool_;
  std::unique_ptr<NaiveVkBufferAllocator> memory_pool_{nullptr};
  VkBufferWithSize root_buffer_;
  VkBufferWithSize global_tmps_buffer_;
  VkBufferWithSize fake_ctx_buffer_;

  std::vector<std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;
  std::vector<std::string> pending_kernels_;
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
