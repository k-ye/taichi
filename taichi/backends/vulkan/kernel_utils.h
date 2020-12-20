#pragma once

#include <optional>
#include <string>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/type.h"

TLANG_NAMESPACE_BEGIN

class Kernel;
class SNode;

namespace vulkan {

struct TaskAttributes {
  enum class Buffers {
    Root,
    GlobalTmps,
    Context,
  };

  struct BufferBind {
    Buffers type;
    int binding;
  };

  std::string name;
  // Total number of threads to launch (i.e. threads per grid). Note that this
  // is only advisory, because eventually this number is also determined by the
  // runtime config. This works because grid strided loop is supported.
  int advisory_total_num_threads;
  int advisory_num_threads_per_group;

  OffloadedTaskType task_type;

  struct RangeForAttributes {
    // |begin| has differen meanings depending on |const_begin|:
    // * true : It is the left boundary of the loop known at compile time.
    // * false: It is the offset of the begin in the global tmps buffer.
    //
    // Same applies to |end|.
    size_t begin;
    size_t end;
    bool const_begin{true};
    bool const_end{true};

    inline bool const_range() const {
      return (const_begin && const_end);
    }
  };
  std::vector<BufferBind> buffer_binds;
  // Only valid when |task_type| is range_for.
  std::optional<RangeForAttributes> range_for_attribs;

  static std::string buffers_name(Buffers b);
  std::string debug_string() const;
};

// This class contains the attributes descriptors for both the input args and
// the return values of a Taichi kernel.
//
// Note that all Vulkan tasks (shaders) belonging to the same Taichi kernel will
// share the same kernel args (i.e. they use the same Vulkan buffer for input
// args and return values). This is because kernel arguments is a Taichi-level
// concept.
class KernelContextAttributes {
 private:
  // Attributes that are shared by the input arg and the return value.
  struct AttribsBase {
    // For scalar arg, this is max(stride(dt), 4)
    // For array arg, this is #elements * max(stride(dt), 4)
    // Unit: byte
    size_t stride = 0;
    // Offset in the context buffer
    size_t offset_in_mem = 0;
    // Index of the input arg or the return value in the host `Context`
    int index = -1;
    DataType dt;
    bool is_array = false;
  };

 public:
  // This is mostly the same as Kernel::Arg, with Vulkan specific attributes.
  struct ArgAttributes : public AttribsBase {};

  // This is mostly the same as Kernel::Ret, with Vulkan specific attributes.
  struct RetAttributes : public AttribsBase {};

  explicit KernelContextAttributes(const Kernel &kernel);

  inline bool has_args() const {
    return !arg_attribs_vec_.empty();
  }

  inline const std::vector<ArgAttributes> &args() const {
    return arg_attribs_vec_;
  }

  inline bool has_rets() const {
    return !ret_attribs_vec_.empty();
  }

  inline const std::vector<RetAttributes> &rets() const {
    return ret_attribs_vec_;
  }

  // Returns true if the kernel has neither input args nor return values.
  inline bool empty() const {
    return !(has_args() || has_rets());
  }

  // Total size in bytes of the input args and return values
  inline size_t ctx_bytes() const {
    return ctx_bytes_;
  }
  inline size_t extra_args_bytes() const {
    return extra_args_bytes_;
  }
  // Total bytes needed for allocating the Vulkan buffer
  inline size_t total_bytes() const {
    return ctx_bytes_ + extra_args_bytes_;
  }

 private:
  // Memory layout
  //
  // /---- input args ----\/---- ret vals -----\/-- extra args --\
  // +----------+---------+----------+---------+-----------------+
  // |  scalar  |  array  |  scalar  |  array  |      scalar     |
  // +----------+---------+----------+---------+-----------------+
  //
  std::vector<ArgAttributes> arg_attribs_vec_;
  std::vector<RetAttributes> ret_attribs_vec_;
  // Total size in bytes of the input args and return values
  size_t ctx_bytes_;
  size_t extra_args_bytes_;
};

// Groups all the Vulkan kernels generated from a single ti.kernel
struct TaichiKernelAttributes {
  // Taichi kernel name
  std::string name;
  // Is this kernel for evaluating the constant fold result?
  bool is_jit_evaluator = false;
  // Attributes of all the tasks produced from this single Taichi kernel.
  std::vector<TaskAttributes> tasks_attribs;

  std::optional<KernelContextAttributes> ctx_attribs = std::nullopt;
};

}  // namespace vulkan
TLANG_NAMESPACE_END