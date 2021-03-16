#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <optional>
#include <vector>

namespace taichi {
namespace lang {
namespace vulkan {

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {}
};

struct VulkanQueueFamilyIndices {
  std::optional<uint32_t> compute_family;
  // Do we also need a transfer_family?

  bool is_complete() const { return compute_family.has_value(); }
};

class VulkanDevice {
 public:
  struct Params {
    uint32_t api_version{VK_API_VERSION_1_0};
  };
  explicit VulkanDevice(const Params &params);
  ~VulkanDevice();

  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  const VulkanQueueFamilyIndices &queue_family_indices() const {
    return queue_family_indices_;
  }
  VkQueue compute_queue() const { return compute_queue_; }
  VkCommandPool command_pool() const { return command_pool_; }

 private:
  void create_instance(const Params &params);
  void setup_debug_messenger();
  void pick_physical_device();
  void create_logical_device();
  void create_command_pool();

  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VulkanQueueFamilyIndices queue_family_indices_;
  VkDevice device_{VK_NULL_HANDLE};
  // TODO: It's probably not right to put these per-queue things here. However,
  // in Taichi we only use a single queue on a single device (i.e. a single CUDA
  // stream), so it doesn't make a difference.
  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkCommandPool command_pool_{VK_NULL_HANDLE};
};

class VulkanPipeline {
 public:
  struct BufferBinding {
    VkBuffer buffer{VK_NULL_HANDLE};
    uint32_t binding{0};
  };

  struct Params {
    const VulkanDevice *device{nullptr};
    std::vector<BufferBinding> buffer_bindings;
    SpirvCodeView code;
  };

  explicit VulkanPipeline(const Params &params);
  ~VulkanPipeline();

  VkPipelineLayout pipeline_layout() const { return pipeline_layout_; }
  VkPipeline pipeline() const { return pipeline_; }
  const VkDescriptorSet &descriptor_set() const { return descriptor_set_; }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_compute_pipeline(const Params &params);
  void create_descriptor_pool(const Params &params);
  void create_descriptor_sets(const Params &params);

  VkDevice device_;  // not owned

  // TODO: Commands using the same Taichi buffers should be able to share the
  // same descriptor set layout?
  VkDescriptorSetLayout descriptor_set_layout_;
  // TODO: Commands having the same |descriptor_set_layout_| should be able to
  // share the same pipeline layout?
  VkPipelineLayout pipeline_layout_;
  // This maps 1:1 to a shader, so it needs to be created per compute
  // shader.
  VkPipeline pipeline_;
  VkDescriptorPool descriptor_pool_;
  VkDescriptorSet descriptor_set_;
};

class VulkanCommandBuilder {
 public:
  explicit VulkanCommandBuilder(const VulkanDevice *device);

  ~VulkanCommandBuilder();

  void append(const VulkanPipeline &pipeline, int group_count_x);

  VkCommandBuffer build();

 private:
  // VkCommandBuffers are destroyed when the underlying command pool is
  // destroyed.
  // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers#page_Command-buffer-allocation
  VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
};

class VulkanStream {
 public:
  VulkanStream(const VulkanDevice *device);

  void launch(VkCommandBuffer command);
  void synchronize();

 private:
  const VulkanDevice *const device_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
