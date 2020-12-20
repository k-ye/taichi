#include "taichi/backends/vulkan/runtime.h"

#include <vulkan/vulkan.h>

#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/math/arithmetic.h"

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
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif  // #ifdef NDEBUG

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
  static constexpr VkDeviceSize kAlignment = 16;

  struct Params {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkMemoryPropertyFlags requiredProperties;
    VkDeviceSize poolSize = 0;
  };

  LinearVkMemoryPool(const Params &params, VkDeviceMemory mem, uint32_t mti)
      : device_(params.device),
        memory_(mem),
        memory_type_index_(mti),
        pool_size_(params.poolSize),
        next_(0) {
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

  std::unique_ptr<VkBufferWithMemory> alloc_and_bind(
      VkBufferCreateInfo createInfo) {
    createInfo.size = roundup_aligned(createInfo.size);
    VkBuffer buffer;
    BAIL_ON_VK_BAD_RESULT(
        vkCreateBuffer(device_, &createInfo, kNoVkAllocCallbacks, &buffer),
        "failed to create buffer");

    const auto offset_in_mem = next_;
    next_ += createInfo.size;
    BAIL_ON_VK_BAD_RESULT(
        vkBindBufferMemory(device_, buffer, memory_, offset_in_mem),
        "failed to bind buffer to memory");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);
    TI_ASSERT(memRequirements.memoryTypeBits & (1 << memory_type_index_));
    TI_ASSERT((createInfo.size % memRequirements.alignment) == 0);

    return std::make_unique<VkBufferWithMemory>(device_, buffer, memory_,
                                                createInfo.size, offset_in_mem);
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
      : device_(params.device), computeQueue_(params.compute_queue) {
    CreateDescriptorSetLayout(params);
    CreateComputePipeline(params);
    CreateDescriptorPool(params);
    CreateDescriptorSets(params);
    CreateCommandBuffer(params);
  }

  ~UserVulkanKernel() {
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
    BAIL_ON_VK_BAD_RESULT(vkQueueSubmit(computeQueue_, /*submitCount=*/1,
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
    shaderStageInfo.pName = params.attribs->name.c_str();

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

    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
        /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptorSet_,
        /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/nullptr);
    vkCmdDispatch(commandBuffer_, params.attribs->advisory_total_num_threads,
                  /*groupCountY=*/1,
                  /*groupCountZ=*/1);
    BAIL_ON_VK_BAD_RESULT(vkEndCommandBuffer(commandBuffer_),
                          "failed to record command buffer");
  }

  VkDevice device_;       // not owned
  VkQueue computeQueue_;  // not owned
  VkDescriptorSetLayout descriptorSetLayout_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline pipeline_;
  VkDescriptorPool descriptorPool_;
  VkDescriptorSet descriptorSet_;
  VkCommandBuffer commandBuffer_;
};

}  // namespace

class VkRuntime ::Impl {
 public:
  explicit Impl(const Params &params) {
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
      decltype(vk_kernels_) tmp;
      tmp.swap(vk_kernels_);
    }
    globalTmpsBuffer_.reset();
    rootBuffer_.reset();
    memoryPool_.reset();
    if constexpr (kEnableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_,
                                    kNoVkAllocCallbacks);
    }
    vkDestroyCommandPool(device_, commandPool_, kNoVkAllocCallbacks);
    vkDestroyDevice(device_, kNoVkAllocCallbacks);
    vkDestroyInstance(instance_, kNoVkAllocCallbacks);
  }

  KernelHandle register_taichi_kernel(const TaskAttributes &attribs,
                                      const SpirvCodeView &code) {
    InputBuffersMap input_buffers = {
        {BufferEnum::Root, rootBuffer_.get()},
        {BufferEnum::GlobalTmps, globalTmpsBuffer_.get()},
    };
    UserVulkanKernel::Params params;
    params.code = code;
    params.attribs = &attribs;
    params.input_buffers = &input_buffers;
    params.device = device_;
    params.compute_queue = computeQueue_;
    params.command_pool = commandPool_;

    KernelHandle res;
    res.id_ = vk_kernels_.size();
    vk_kernels_.push_back(std::make_unique<UserVulkanKernel>(params));
    return res;
  }

  void launch_kernel(KernelHandle handle) {
    vk_kernels_[handle.id_]->launch();
  }

  void synchronize() {
    vkQueueWaitIdle(computeQueue_);
  }

  VkBufferWithMemory *root_buffer() {
    return rootBuffer_.get();
  }

  VkBufferWithMemory *global_tmps_buffer() {
    return globalTmpsBuffer_.get();
  }

 private:
  void CreateInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Taichi Vulkan Backend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;  // important

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if constexpr (kEnableValidationLayers) {
      TI_ASSERT_INFO(CheckValidationLayerSupport(),
                     "validation layers requested but not available");
    }
    for (const auto &ext : GetAllExtensionProperties()) {
      std::cout << "ext=" << ext.extensionName << " spec=" << ext.specVersion
                << std::endl;
    }
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
                     /*queueIndex=*/0, &computeQueue_);
  }

  void CreateCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.queueFamilyIndex = queueFamilyIndices_.computeFamily.value();
    BAIL_ON_VK_BAD_RESULT(
        vkCreateCommandPool(device_, &poolInfo, kNoVkAllocCallbacks,
                            &commandPool_),
        "failed to create command pool");
  }

  void init_memory_pool(const Params &params) {
    LinearVkMemoryPool::Params mp_params;
    mp_params.physicalDevice = physicalDevice_;
    mp_params.device = device_;
    /*mp_params.poolSize =
        (params.config->device_memory_GB * 1024 * 1024 * 1024ULL);*/
    mp_params.poolSize = 100 * 1024 * 1024;
    mp_params.requiredProperties = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memoryPool_ = LinearVkMemoryPool::try_make(mp_params);
    TI_ASSERT(memoryPool_ != nullptr);
  }

  void init_vk_buffers() {
    const uint32_t computeQueueFamiltyIndex =
        queueFamilyIndices_.computeFamily.value();
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = 1024 * 1024;
    createInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 1;
    createInfo.pQueueFamilyIndices = &computeQueueFamiltyIndex;
    rootBuffer_ = memoryPool_->alloc_and_bind(createInfo);
    globalTmpsBuffer_ = memoryPool_->alloc_and_bind(createInfo);
  }

  VkInstance instance_;
  VkDebugUtilsMessengerEXT debugMessenger_;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  QueueFamilyIndices queueFamilyIndices_;
  VkDevice device_;
  VkQueue computeQueue_;
  VkCommandPool commandPool_;

  std::unique_ptr<LinearVkMemoryPool> memoryPool_;
  std::unique_ptr<VkBufferWithMemory> rootBuffer_;
  std::unique_ptr<VkBufferWithMemory> globalTmpsBuffer_;

  std::vector<std::unique_ptr<UserVulkanKernel>> vk_kernels_;
};

#else

class VkRuntime::Impl {
 public:
  Impl(const Params &) {
    TI_ERROR("Vulkan disabled");
  }

  KernelHandle register_taichi_kernel(const TaskAttributes &,
                                      const SpirvCodeView &) {
    TI_ERROR("Vulkan disabled");
    return KernelHandle();
  }

  void launch_kernel(KernelHandle) {
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
    const TaskAttributes &attribs,
    const SpirvCodeView &code) {
  return impl_->register_taichi_kernel(attribs, code);
}

void VkRuntime::launch_kernel(KernelHandle handle) {
  impl_->launch_kernel(handle);
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