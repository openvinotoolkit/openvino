// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "passes/check_cycles.hpp"

#include <unordered_map>

#include "util/assert.hpp"
#include "util/map_range.hpp"

#include "graph.hpp"
#include "node.hpp"

namespace ade
{
namespace passes
{
enum class TraverseState
{
    visiting,
    visited,
};

using state_t = std::unordered_map<Node*, TraverseState>;

static void visit(state_t& state, const NodeHandle& node)
{
    ASSERT(nullptr != node);
    state[node.get()] = TraverseState::visiting;
    for (auto adj:
         util::map(node->outEdges(), [](const EdgeHandle& e) { return e->dstNode(); }))
    {
        auto it = state.find(adj.get());
        if (state.end() == it) // not visited
        {
            visit(state, adj);
        }
        else if (TraverseState::visiting == it->second)
        {
            throw_error(CycleFound());
        }
    }
    state[node.get()] = TraverseState::visited;

}

void CheckCycles::operator()(const PassContext& context) const
{
    state_t state;
    for (auto node: context.graph.nodes())
    {
        if (state.end() == state.find(node.get()))
        {
            // not yet visited during recursion
            visit(state, node);
        }
    }
}

std::string CheckCycles::name()
{
    return "CheckCycles";
}

const char* CycleFound::what() const noexcept
{
    return "Cycle was detected in graph";
}

}
}
