// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "passes/topological_sort.hpp"

#include <vector>
#include <unordered_set>

#include "graph.hpp"

namespace ade
{
namespace passes
{
using sorted_t = std::vector<NodeHandle>;
using visited_t = std::unordered_set<Node*>;

static void visit(sorted_t& sorted, visited_t& visited, const NodeHandle& node)
{
    if (visited.end() == visited.find(node.get()))
    {
        for (auto adj:
             util::map(node->inEdges(), [](const EdgeHandle& e) { return e->srcNode(); }))
        {
            visit(sorted, visited, adj);
        }
        sorted.push_back(node);
        visited.insert(node.get());
    }
}

void TopologicalSort::operator()(TypedPassContext<TopologicalSortData> context) const
{
    sorted_t sorted;
    visited_t visited;
    for (auto node: context.graph.nodes())
    {
        visit(sorted, visited, node);
    }
    context.graph.metadata().set(TopologicalSortData(std::move(sorted)));
}

const char* TopologicalSort::name()
{
    return "TopologicalSort";
}

const char* TopologicalSortData::name()
{
    return "TopologicalSortData";
}

bool LazyTopologicalSortChecker::nodeCreated(const Graph& /*graph*/, const NodeHandle& /*node*/)
{
    // We need to rebuild nodes list after nodes creation
    return false;
}

bool LazyTopologicalSortChecker::nodeAboutToBeDestroyed(const Graph& /*graph*/, const NodeHandle& /*node*/)
{
    // Removing node cannot change topological order and sorter nodes list can correctly handle nodes removal
    return true;
}

bool LazyTopologicalSortChecker::edgeCreated(const Graph& /*graph*/, const EdgeHandle& /*edge*/)
{
    // Adding edge CAN change topological order
    return false;
}

bool LazyTopologicalSortChecker::edgeAboutToBeDestroyed(const Graph& /*graph*/, const EdgeHandle& /*edge*/)
{
    // Removing edge cannot change topological order
    return true;
}

bool LazyTopologicalSortChecker::edgeAboutToBeRelinked(const Graph& /*graph*/,
                                                       const EdgeHandle& /*edge*/,
                                                       const NodeHandle& /*newSrcNode*/,
                                                       const NodeHandle& /*newDstNode*/)
{
    // Relinking edge CAN change topological order
    return false;
}

}
}
