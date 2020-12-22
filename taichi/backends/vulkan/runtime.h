#pragma once
#include "taichi/lang_util.h"

#include <vector>

#include "taichi/backends/vulkan/snode_struct_compiler.h"
#include "taichi/backends/vulkan/kernel_utils.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#include "taichi/program/compile_config.h"

TLANG_NAMESPACE_BEGIN
namespace vulkan {

class VkRuntime {
 private:
  class Impl;

 public:
  struct Params {
    // CompiledSNodeStructs compiled_snode_structs;
    const CompileConfig *config;
    // uint64_t *host_result_buffer;
    // int root_id;
    const SNodeDescriptorsMap *snode_descriptors = nullptr;
  };

  explicit VkRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~VkRuntime();

  class KernelHandle {
   private:
    friend class Impl;
    int id_ = -1;
  };

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<std::string> task_glsl_source_codes;
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  void launch_kernel(KernelHandle handle);

  void synchronize();

  VkBufferWithMemory *root_buffer();
  VkBufferWithMemory *global_tmps_buffer();

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace vulkan

TLANG_NAMESPACE_END