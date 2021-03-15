#pragma once

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <vector>
#include <optional>
#include <vector>

namespace taichi {
namespace lang {
namespace vulkan {
#if 0
class VkBufferWithMemory {
 public:
  VkBufferWithMemory(VkDevice device,
                     VkBuffer buffer,
                     VkDeviceMemory mem,
                     VkDeviceSize size,
                     VkDeviceSize offset)
      : device_(device),
        buffer_(buffer),
        backing_memory_(mem),
        size_(size),
        offset_in_mem_(offset) {
    TI_ASSERT(buffer_ != VK_NULL_HANDLE);
    TI_ASSERT(size_ > 0);
    TI_ASSERT(backing_memory_ != VK_NULL_HANDLE);

    vkMapMemory(device_, backing_memory_, offset_in_mem_, size_, /*flags=*/0,
                &data_);
  }

  // Just use std::unique_ptr to save all the trouble from crafting move ctors
  // on our own
  VkBufferWithMemory(const VkBufferWithMemory &) = delete;
  VkBufferWithMemory &operator=(const VkBufferWithMemory &) = delete;
  VkBufferWithMemory(VkBufferWithMemory &&) = delete;
  VkBufferWithMemory &operator=(VkBufferWithMemory &&) = delete;

  ~VkBufferWithMemory() {
    if (buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer_, nullptr);
    }
  }

  VkBuffer buffer() const {
    return buffer_;
  }

  VkDeviceSize size() const {
    return size_;
  }

  VkDeviceSize offset_in_mem() const {
    return offset_in_mem_;
  }

  class Mapped {
   public:
    explicit Mapped(void *data) : data_(data) {
    }

    void *data() const {
      return data_;
    }

   private:
    // VkBufferWithMemory *const buf_;  // not owned
    void *data_;
  };

  Mapped map_mem() {
    return Mapped(data_);
  }

 private:
  friend class VkBufferWithMemory;
  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory backing_memory_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
  VkDeviceSize offset_in_mem_ = 0;
  void *data_;
};
#endif
struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {
  }
};

std::vector<VkExtensionProperties> GetInstanceExtensionProperties();

std::vector<VkExtensionProperties> GetDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice);

class VulkanEnvSettings {
 public:
  static constexpr uint32_t kApiVersion() {
    return VK_API_VERSION_1_0;
  }

  static constexpr shaderc_env_version kShadercEnvVersion() {
    return shaderc_env_version_vulkan_1_0;
  }
};

class GlslToSpirvCompiler {
 public:
  explicit GlslToSpirvCompiler();

  using SpirvBinary = std::vector<uint32_t>;
  std::optional<SpirvBinary> compile(const std::string &glsl,
                                     const std::string kernel_name);

 private:
  shaderc::CompileOptions opts_;
  shaderc::Compiler compiler_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
