#pragma once

#include "taichi/lang_util.h"

#include "taichi/backends/vulkan/snode_struct_compiler.h"

TLANG_NAMESPACE_BEGIN

class Kernel;

namespace vulkan {

void lower(Kernel *kernel);

// These ASTs must have already been lowered at the CHI level.
FunctionType compile_to_executable(
    Kernel *kernel,
    const CompiledSNodeStructs *compiled_structs);

}  // namespace vulkan

TLANG_NAMESPACE_END