#pragma once

#include <unordered_set>

namespace taichi {
namespace lang {

class SNode;

/**
 *
 */
class VisitedSNodesCache {
 public:
  /**
   * @param snode:
   * @returns
   */
  bool has_visited(const SNode *snode) const {
    return visited_.count(snode) > 0;
  }

  /**
   * @param snode:
   * @returns
   */
  void put(const SNode *snode) {
    visited_.insert(snode);
  }

 private:
  std::unordered_set<const SNode *> visited_;
};

}  // namespace lang
}  // namespace taichi