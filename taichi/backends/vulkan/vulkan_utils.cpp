#include "taichi/backends/vulkan/vulkan_utils.h"

TLANG_NAMESPACE_BEGIN
namespace vulkan {

std::vector<VkExtensionProperties> GetAllExtensionProperties() {
  constexpr char *kNoLayerName = nullptr;
  uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(kNoLayerName, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateInstanceExtensionProperties(kNoLayerName, &count,
                                         extensions.data());
  return extensions;
}

}  // namespace vulkan
TLANG_NAMESPACE_END