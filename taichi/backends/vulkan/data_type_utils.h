#pragma once

#include <algorithm>
#include <cstddef>

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace vulkan {

inline std::size_t vk_data_type_size(DataType dt) {
  // Vulkan buffers require a minimum alignment of 4 bytes.
  // https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment-requirements
  return std::max(data_type_size(dt), 4);
}

}  // namespace vulkan
TLANG_NAMESPACE_END