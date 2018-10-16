// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <unordered_set>
#include <unordered_map>

#include <node.hpp>

#include <util/algorithm.hpp>
#include <util/assert.hpp>
#include <util/func_ref.hpp>

namespace ade
{
namespace traverse
{
using traverse_func_type
    = util::func_ref<void (const Node&,
                           util::func_ref<void (const NodeHandle&)>)>;
inline void forward(const Node& node,
                    util::func_ref<void (const NodeHandle&)> visitor)
{
    for (auto&& next : node.outNodes())
    {
        visitor(next);
    }
}
inline void backward(const Node& node,
                     util::func_ref<void (const NodeHandle&)> visitor)
{
    for (auto&& next : node.inNodes())
    {
        visitor(next);
    }
}
} // namespace ade::traverse

/// Depth first search through node output edges
///
/// @param node - Start node, must not be null
/// @param visitor - Functor called for each found node,
/// can return true to continue search though found node output edges
void dfs(const NodeHandle& node,
         util::func_ref<bool (const NodeHandle&)> visitor,
         traverse::traverse_func_type direction = traverse::forward);

namespace details
{
struct TransitiveClosureHelper
{
    using CacheT =
    std::unordered_map<NodeHandle,
                       std::unordered_set<NodeHandle, HandleHasher<Node>>,
                       HandleHasher<Node>>;
    void operator()(CacheT& cache,
                    const NodeHandle& node,
                    traverse::traverse_func_type direction) const;
};
}

/// Enumerate all nodes reachable through outputs for each source node in
/// provided list
///
/// @param nodes - List of nodes to check
/// @param visitor - functor which will be called for pairs of nodes,
/// first parameted is source node from provided list and
/// second parameter is node reachable from this source node
template<typename Nodes, typename Visitor>
void transitiveClosure(
        Nodes&& nodes,
        Visitor&& visitor,
        traverse::traverse_func_type direction = traverse::forward)
{
    using Helper = details::TransitiveClosureHelper;
    Helper::CacheT visited;
    for (auto node : nodes)
    {
        ASSERT(nullptr != node);
        if (!util::contains(visited, node))
        {
            Helper()(visited, node, direction);
        }
        ASSERT(util::contains(visited, node));
        for (auto nextNode : visited[node])
        {
            visitor(node, nextNode);
        }
    }
}

}

#endif // SEARCH_HPP
