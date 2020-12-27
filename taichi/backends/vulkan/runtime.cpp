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

#include "taichi/math/arithmetic.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

namespace vulkan {

#define TI_WITH_VULKAN
#ifdef TI_WITH_VULKAN

#define BAIL_ON_VK_BAD_RESULT(result, msg)        \
  do {                                            \
    TI_ERROR_IF(((result) != VK_SUCCESS), (msg)); \
  } while (0)

namespace {

/// <summary>
///  Vulkan helper API
/// </summary>
constexpr VkAllocationCallbacks *kNoVkAllocCallbacks = nullptr;
#ifdef NDEBUG
constexpr bool kEnableValidationLayers = true;
#else
constexpr bool kEnableValidationLayers = true;
#endif  // #ifdef NDEBUG

// constexpr std::array<const char *, 1> kValidationLayers = {
//     "VK_LAYER_KHRONOS_validation",
// };
const std::vector<const char *> kValidationLayers = {};

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
  // if constexpr (kEnableValidationLayers) {
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  // }
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
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  constexpr VkQueueFlags kFlagMask =
      (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT));

  // first try and find a queue that has just the compute bit set
  for (int i = 0; i < (int)queueFamilyCount; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queueFamilies[i].queueFlags;
    if ((masked_flags & VK_QUEUE_COMPUTE_BIT) &&
        !(masked_flags & VK_QUEUE_GRAPHICS_BIT)) {
      indices.computeFamily = i;
    }
    if (indices.IsComplete()) {
      return indices;
    }
  }

  // lastly get any queue that will work
  for (int i = 0; i < (int)queueFamilyCount; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queueFamilies[i].queueFlags;
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
  return FindQueueFamilies(device).IsComplete();
}

// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/custom_memory_pools.html
class LinearVkMemoryPool {
 public:
  static constexpr VkDeviceSize kAlignment = 256;

  struct Params {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkMemoryPropertyFlags requiredProperties;
    VkDeviceSize poolSize = 0;
    uint32_t computeQueueFamilyIndex = 0;
    VkBufferCreateInfo bufferCreationTemplate;
  };

  LinearVkMemoryPool(const Params &params, VkDeviceMemory mem, uint32_t mti)
      : device_(params.device),
        memory_(mem),
        memory_type_index_(mti),
        compute_queue_family_index_(params.computeQueueFamilyIndex),
        buffer_creation_template_(params.bufferCreationTemplate),
        pool_size_(params.poolSize),
        next_(0) {
    buffer_creation_template_.size = 0;
    buffer_creation_template_.queueFamilyIndexCount = 1;
    buffer_creation_template_.pQueueFamilyIndices =
        &compute_queue_family_index_;
  }

  ~LinearVkMemoryPool() {
    if (memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory_, kNoVkAllocCallbacks);
    }
  }

  static std::unique_ptr<LinearVkMemoryPool> try_make(Params params) {
    params.poolSize = roundup_aligned(params.poolSize);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = params.poolSize;
    const auto memTypeIndex = FindMemoryType(params);
    if (!memTypeIndex.has_value()) {
      return nullptr;
    }
    allocInfo.memoryTypeIndex = memTypeIndex.value();
    VkDeviceMemory mem;
    if (vkAllocateMemory(params.device, &allocInfo, kNoVkAllocCallbacks,
                         &mem) != VK_SUCCESS) {
      return nullptr;
    }
    return std::make_unique<LinearVkMemoryPool>(params, mem,
                                                allocInfo.memoryTypeIndex);
  }

  std::unique_ptr<VkBufferWithMemory> alloc_and_bind(VkDeviceSize buf_size) {
    buf_size = roundup_aligned(buf_size);
    if (pool_size_ <= (next_ + buf_size)) {
      TI_WARN("Vulkan memory pool exhausted, max size={}", pool_size_);
      return nullptr;
    }

    VkBuffer buffer;
    buffer_creation_template_.size = buf_size;
    BAIL_ON_VK_BAD_RESULT(vkCreateBuffer(device_, &buffer_creation_template_,
                                         kNoVkAllocCallbacks, &buffer),
                          "failed to create buffer");
    buffer_creation_template_.size = 0;  // reset
    const auto offset_in_mem = next_;
    next_ += buf_size;
    BAIL_ON_VK_BAD_RESULT(
        vkBindBufferMemory(device_, buffer, memory_, offset_in_mem),
        "failed to bind buffer to memory");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);
    TI_ASSERT(memRequirements.memoryTypeBits & (1 << memory_type_index_));
    TI_ASSERT_INFO((buf_size % memRequirements.alignment) == 0,
                   "buf_size={} required alignment={}", buf_size,
                   memRequirements.alignment);
    return std::make_unique<VkBufferWithMemory>(device_, buffer, memory_,
                                                buf_size, offset_in_mem);
  }

 private:
  static VkDeviceSize roundup_aligned(VkDeviceSize size) {
    return iroundup(size, kAlignment);
  }

  static std::optional<uint32_t> FindMemoryType(const Params &params) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(params.physicalDevice, &memProperties);
    auto satisfies = [&](int i) -> bool {
      const auto &memType = memProperties.memoryTypes[i];
      if ((memType.propertyFlags & params.requiredProperties) !=
          params.requiredProperties) {
        return false;
      }
      if (memProperties.memoryHeaps[memType.heapIndex].size <=
          params.poolSize) {
        return false;
      }
      return true;
    };

    for (int i = 0; i < memProperties.memoryTypeCount; ++i) {
      if (satisfies(i)) {
        return i;
      }
    }
    return std::nullopt;
  }

  VkDevice device_ = VK_NULL_HANDLE;  // not owned
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  uint32_t memory_type_index_ = 0;
  uint32_t compute_queue_family_index_ = 0;
  VkBufferCreateInfo buffer_creation_template_;
  VkDeviceSize pool_size_ = 0;
  VkDeviceSize next_ = 0;
};

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

/// Taichi Helpers

using BufferEnum = TaskAttributes::Buffers;
using InputBuffersMap = std::unordered_map<BufferEnum, VkBufferWithMemory *>;

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
    CreateSyncObjects();
  }

  ~UserVulkanKernel() {
    vkDestroyFence(device_, sync_fence_, kNoVkAllocCallbacks);
    vkDestroyDescriptorPool(device_, descriptorPool_, kNoVkAllocCallbacks);
    vkDestroyPipeline(device_, pipeline_, kNoVkAllocCallbacks);
    vkDestroyPipelineLayout(device_, pipelineLayout_, kNoVkAllocCallbacks);
    vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_,
                                 kNoVkAllocCallbacks);
  }

  void launch() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;

    // auto pfnQueueDebugBegin =
    //     (PFN_vkQueueBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(
    //         device_, "vkQueueBeginDebugUtilsLabelEXT");
    // TI_ASSERT_INFO(pfnQueueDebugBegin != nullptr,
    //                "Cannot find vkQueueBeginDebugUtilsLabelEXT");
    // auto pfnQueueDebugEnd =
    //     (PFN_vkQueueEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(
    //         device_, "vkQueueEndDebugUtilsLabelEXT");
    // TI_ASSERT_INFO(pfnQueueDebugEnd != nullptr,
    //                "Cannot find vkQueueEndDebugUtilsLabelEXT");
    // VkDebugUtilsLabelEXT debugLabelInfo{};
    // debugLabelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    // debugLabelInfo.pLabelName = "fkldsajflksajd";
    // debugLabelInfo.color[1] = 1.0;
    // debugLabelInfo.color[3] = 1.0f;

    // pfnQueueDebugBegin(compute_queue_, &debugLabelInfo);
    StopWatch sw;
    BAIL_ON_VK_BAD_RESULT(vkQueueSubmit(compute_queue_, /*submitCount=*/1,
                                        &submitInfo, /*fence=*/VK_NULL_HANDLE),
                          "failed to submit command buffer");
    // pfnQueueDebugEnd(compute_queue_);
    // vkWaitForFences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX);
    // vkResetFences(device_, 1, &sync_fence_);
    vkQueueWaitIdle(compute_queue_);
    auto dur = sw.GetMicros();

    TI_INFO("running {} took {} us", name_, dur);
    TI_INFO("<<<<<<<<<<<<<<<<<<<<<");
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
      auto *bm = params.input_buffers->at(bb.type);
      VkDescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = bm->buffer();
      // Note that this is the offset within the buffer itself, not the offset
      // of this buffer within its backing memory!
      bufferInfo.offset = 0;
      bufferInfo.range = bm->size();
      descriptorBufferInfos.push_back(bufferInfo);
    }

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

    auto pfnCmdDebugBegin =
        (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(
            device_, "vkCmdBeginDebugUtilsLabelEXT");
    TI_ASSERT_INFO(pfnCmdDebugBegin != nullptr,
                   "Cannot find vkCmdBeginDebugUtilsLabelEXT");
    auto pfnCmdDebugEnd = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(
        device_, "vkCmdEndDebugUtilsLabelEXT");
    TI_ASSERT_INFO(pfnCmdDebugEnd != nullptr,
                   "Cannot find vkCmdEndDebugUtilsLabelEXT");
    auto pfnCmdDebugInsert =
        (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetDeviceProcAddr(
            device_, "vkCmdInsertDebugUtilsLabelEXT");
    TI_ASSERT_INFO(pfnCmdDebugInsert != nullptr,
                   "Cannot find vkCmdInsertDebugUtilsLabelEXT");

    const auto *attribs = params.attribs;
    VkDebugUtilsLabelEXT debugLabelInfo{};
    debugLabelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    debugLabelInfo.pLabelName = attribs->name.c_str();
    pfnCmdDebugBegin(commandBuffer_, &debugLabelInfo);
    pfnCmdDebugInsert(commandBuffer_, &debugLabelInfo);
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
        /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptorSet_,
        /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/nullptr);
    const auto group_x = attribs->advisory_total_num_threads /
                         attribs->advisory_num_threads_per_group;
    // const auto group_x = attribs->advisory_total_num_threads;
    TI_INFO(
        "created buffer for kernel={} total_num={} local_group_size={} "
        "num_group_x={}",
        attribs->name, attribs->advisory_total_num_threads,
        attribs->advisory_num_threads_per_group, group_x);
    vkCmdDispatch(commandBuffer_, group_x,
                  /*groupCountY=*/1,
                  /*groupCountZ=*/1);
    pfnCmdDebugEnd(commandBuffer_);
    BAIL_ON_VK_BAD_RESULT(vkEndCommandBuffer(commandBuffer_),
                          "failed to record command buffer");
  }

  void CreateSyncObjects() {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = 0;
    BAIL_ON_VK_BAD_RESULT(
        vkCreateFence(device_, &fence_info, nullptr, &sync_fence_),
        "failed to create sync fence");
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
  VkFence sync_fence_;
};

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
    StopWatch sw;
    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg = ctx_attribs_->args()[i];
      char *device_ptr = device_base + arg.offset_in_mem;
      sw.GetMicros();
      if (arg.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, arg.stride);
        auto dur = sw.GetMicros();
        // TI_INFO("D2H arg array i={} duration={} us", i, dur);
      }
    }
    for (int i = 0; i < ctx_attribs_->rets().size(); ++i) {
      // Note that we are copying the i-th return value on Metal to the i-th
      // *arg* on the host context.
      const auto &ret = ctx_attribs_->rets()[i];
      char *device_ptr = device_base + ret.offset_in_mem;
      const auto dt = ret.dt;
      sw.GetMicros();

      if (ret.is_array) {
        void *host_ptr = host_ctx_->get_arg<void *>(i);
        std::memcpy(host_ptr, device_ptr, ret.stride);
        auto dur = sw.GetMicros();
        // TI_INFO("D2H ret array i={} duration={} us", i, dur);
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
// compiled Vulkan kernels.
class CompiledTaichiKernel {
 public:
  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs = nullptr;
    std::vector<GlslToSpirvCompiler::SpirvBinary> spirv_bins;
    const SNodeDescriptorsMap *snode_descriptors = nullptr;
    VkBufferWithMemory *root_buffer = nullptr;
    VkBufferWithMemory *global_tmps_buffer = nullptr;
    LinearVkMemoryPool *vk_mem_pool = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
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

  // Have to be exposed as public for Impl to use. We cannot friend the Impl
  // class because it is private.
  TaichiKernelAttributes ti_kernel_attribs;
  std::unique_ptr<VkBufferWithMemory> ctx_buffer_;
  std::vector<std::unique_ptr<UserVulkanKernel>> vk_kernels;
};

}  // namespace

class VkRuntime ::Impl {
 public:
  explicit Impl(const Params &params)
      : config_(params.config),
        snode_descriptors_(params.snode_descriptors),
        host_result_buffer_(params.host_result_buffer) {
    TI_ASSERT(config_ != nullptr);
    TI_ASSERT(snode_descriptors_ != nullptr);
    TI_ASSERT(host_result_buffer_ != nullptr);
    CreateInstance();
    SetupDebugMessenger();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateCommandPool();
    CreateSyncObjects();
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
    memory_pool_.reset();
    if constexpr (kEnableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_,
                                    kNoVkAllocCallbacks);
    }
    vkDestroyFence(device_, sync_fence_, kNoVkAllocCallbacks);
    vkDestroyCommandPool(device_, command_pool_, kNoVkAllocCallbacks);
    vkDestroyDevice(device_, kNoVkAllocCallbacks);
    vkDestroyInstance(instance_, kNoVkAllocCallbacks);
  }

  KernelHandle register_taichi_kernel(RegisterParams reg_params) {
    CompiledTaichiKernel::Params params;
    params.ti_kernel_attribs = &(reg_params.kernel_attribs);
    params.snode_descriptors = snode_descriptors_;
    params.root_buffer = root_buffer_.get();
    params.global_tmps_buffer = global_tmps_buffer_.get();
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
    auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
        &ti_kernel->ti_kernel_attribs.ctx_attribs, host_ctx,
        host_result_buffer_, ti_kernel->ctx_buffer_.get());
    if (ctx_blitter) {
      TI_ASSERT(ti_kernel->ctx_buffer_ != nullptr);
      ctx_blitter->host_to_device();
    }
    int i = 0;
    for (auto &vk : ti_kernel->vk_kernels) {
      vk->launch();
      pending_kernels_.push_back(
          ti_kernel->ti_kernel_attribs.tasks_attribs[i].name);
      ++i;
    }
    if (ctx_blitter) {
      sync(/*for_ctx=*/true);
      ctx_blitter->device_to_host();
    }
  }

  void synchronize() {
    sync(/*for_ctx=*/false);
  }

  VkBufferWithMemory *root_buffer() {
    return root_buffer_.get();
  }

  VkBufferWithMemory *global_tmps_buffer() {
    return global_tmps_buffer_.get();
  }

 private:
  void sync(bool for_ctx) {
    return;

    vkQueueWaitIdle(compute_queue_);
    // BAIL_ON_VK_BAD_RESULT(vkQueueSubmit(compute_queue_, /*submitCount=*/0,
    //                                     nullptr, /*fence=*/sync_fence_),
    //                       "failed to submit dummy sync");
    // vkWaitForFences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX);
    // vkResetFences(device_, 1, &sync_fence_);
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
    // auto props = GetDeviceExtensionProperties(physicalDevice_);
    // for (const auto &p : props) {
    //   std::cout << "  device ext=" << p.extensionName << std::endl;
    // }
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

  void CreateSyncObjects() {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    BAIL_ON_VK_BAD_RESULT(
        vkCreateFence(device_, &fence_info, nullptr, &sync_fence_),
        "failed to create sync fence");
  }

  void init_memory_pool(const Params &params) {
    LinearVkMemoryPool::Params mp_params;
    mp_params.physicalDevice = physicalDevice_;
    mp_params.device = device_;
    /*mp_params.poolSize =
        (params.config->device_memory_GB * 1024 * 1024 * 1024ULL);*/
    mp_params.poolSize = 10 * 1024 * 1024;
    mp_params.requiredProperties = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // mp_params.requiredProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    mp_params.computeQueueFamilyIndex =
        queueFamilyIndices_.computeFamily.value();

    auto &bufTemplate = mp_params.bufferCreationTemplate;
    bufTemplate.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufTemplate.pNext = nullptr;
    bufTemplate.flags = 0;
    bufTemplate.size = 0;
    bufTemplate.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufTemplate.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufTemplate.queueFamilyIndexCount = 1;
    bufTemplate.pQueueFamilyIndices = nullptr;

    memory_pool_ = LinearVkMemoryPool::try_make(mp_params);
    TI_ASSERT(memory_pool_ != nullptr);
  }

  void init_vk_buffers() {
    root_buffer_ = memory_pool_->alloc_and_bind(1024 * 1024);
    global_tmps_buffer_ = memory_pool_->alloc_and_bind(1024 * 1024);
  }

  const CompileConfig *const config_;
  const SNodeDescriptorsMap *const snode_descriptors_;
  uint64_t *const host_result_buffer_;
  VkInstance instance_;
  VkDebugUtilsMessengerEXT debugMessenger_;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  QueueFamilyIndices queueFamilyIndices_;
  VkDevice device_;
  VkQueue compute_queue_;
  VkCommandPool command_pool_;
  VkFence sync_fence_;
  GlslToSpirvCompiler spv_compiler_;

  std::unique_ptr<LinearVkMemoryPool> memory_pool_;
  std::unique_ptr<VkBufferWithMemory> root_buffer_;
  std::unique_ptr<VkBufferWithMemory> global_tmps_buffer_;

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

VkBufferWithMemory *VkRuntime::root_buffer() {
  return impl_->root_buffer();
}

VkBufferWithMemory *VkRuntime::global_tmps_buffer() {
  return impl_->global_tmps_buffer();
}

}  // namespace vulkan
TLANG_NAMESPACE_END