// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"

#include "vpu/utils/error.hpp"

#include <stack>
#include <deque>

namespace vpu {

std::shared_ptr<ngraph::Node> shapeToConstant(const ngraph::element::Type& type, const ngraph::Shape& shape);

std::shared_ptr<ngraph::Node> gatherShapeElements(const ngraph::Output<ngraph::Node>&, int startIndex, size_t elemCount);

std::shared_ptr<ngraph::Node> gatherShapeElements(const ngraph::Output<ngraph::Node>& shape, const std::vector<int64_t>& indicesToGather);

template<>
inline void printTo(std::ostream& stream, const ngraph::NodeTypeInfo& object) {
    stream << object.name << " ver. " << object.version;
}

using Nodes = std::unordered_set<ngraph::Node*>;

template<class GetNext, class Visit>
Nodes dfs(ngraph::Node* root, GetNext&& getNext, Visit&& visit) {
    Nodes visited;
    std::stack<ngraph::Node*> stack{{root}};
    while (!stack.empty()) {
        const auto current = stack.top();
        stack.pop();

        if (!visited.emplace(current).second) {
            continue;
        }

        if (!visit(current)) {
            continue;
        }

        for (const auto& next : getNext(current)) {
            stack.push(next);
        }
    }
    return visited;
}

template<class NumEntries, class Visit, class MoveForward>
void bfs(ngraph::Node* root, NumEntries&& getNumEntries, Visit&& visit, MoveForward&& moveForward) {
    std::deque<ngraph::Node*> deque{root};
    std::unordered_map<ngraph::Node*, std::size_t> visits;
    while (!deque.empty()) {
        const auto current = deque.front();
        deque.pop_front();

        const auto numEntries = current == root ? 1 : getNumEntries(current);

        const auto visitsCount = ++visits[current];
        VPU_THROW_UNLESS(visitsCount <= numEntries, "Encountered loop at {}", current);

        if (visitsCount < numEntries) {
            VPU_THROW_UNLESS(!deque.empty(), "Node {} should be visited only after all predecessors, but it is not available through all of them", current);
            continue;
        }

        if (!visit(current)) {
            continue;
        }

        moveForward(deque, current);
    }
}

}  // namespace vpu
