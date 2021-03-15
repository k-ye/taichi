#pragma once

#include <vector>

#ifndef IN_TAI_VULKAN

#include "taichi/backends/vulkan/kernel_utils.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#include "taichi/program/compile_config.h"

#else

#include "src/utils.h"

#endif  // IN_TAI_VULKAN

namespace taichi {
namespace lang {

struct Context;

namespace vulkan {

class VkRuntime {
 private:
  class Impl;

 public:
  struct Params {
    // CompiledSNodeStructs compiled_snode_structs;
    const CompileConfig *config = nullptr;
    uint64_t *host_result_buffer = nullptr;
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

  void launch_kernel(KernelHandle handle, Context *host_ctx);

  void synchronize();

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
