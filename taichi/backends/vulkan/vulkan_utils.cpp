#include "taichi/backends/vulkan/vulkan_utils.h"

#include <spirv-tools/libspirv.hpp>

#include "taichi/common/logging.h"

namespace taichi {
namespace lang {
namespace vulkan {

std::vector<VkExtensionProperties> GetInstanceExtensionProperties() {
  constexpr char *kNoLayerName = nullptr;
  uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(kNoLayerName, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateInstanceExtensionProperties(kNoLayerName, &count,
                                         extensions.data());
  return extensions;
}

std::vector<VkExtensionProperties> GetDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice) {
  constexpr char *kNoLayerName = nullptr;
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, kNoLayerName, &count,
                                       nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(physicalDevice, kNoLayerName, &count,
                                       extensions.data());
  return extensions;
}

GlslToSpirvCompiler::GlslToSpirvCompiler() {
  opts_.SetTargetEnvironment(shaderc_target_env_vulkan,
                             VulkanEnvSettings::kShadercEnvVersion());
}

std::optional<GlslToSpirvCompiler::SpirvBinary> GlslToSpirvCompiler::compile(
    const std::string &glsl,
    const std::string kernel_name) {
  TI_INFO("Compiling GLSL -> SPIR-V for kernel={}\n{}", kernel_name, glsl);
  auto spv_result =
      compiler_.CompileGlslToSpv(glsl, shaderc_glsl_default_compute_shader,
                                 /*input_file_name=*/kernel_name.c_str(),
                                 /*entry_point_name=*/"main", opts_);
  if (spv_result.GetCompilationStatus() != shaderc_compilation_status_success) {
    TI_WARN("Failed to compile kernel={}, GLSL source:\n{}", kernel_name, glsl);
    TI_ERROR("Compilation errors:\n{}", spv_result.GetErrorMessage());
    return std::nullopt;
  }
  TI_INFO("Succesfully compiled GLSL for kernel={}", kernel_name);
  SpirvBinary res(spv_result.begin(), spv_result.end());
  TI_ASSERT(!res.empty());
  return res;
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
