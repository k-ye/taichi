#pragma once

#include "taichi/struct/visited_snodes_cache.h"

namespace taichi {
namespace lang {

class SNode;

class SNodePropertiesInferrer {
 public:
  explicit SNodePropertiesInferrer(VisitedSNodesCache *cache);

  void infer(SNode *root);

 private:
  void infer_snode_properties(SNode *snode);

  VisitedSNodesCache *const snodes_cache_;
};

}  // namespace lang
}  // namespace taichi