// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

namespace ov::intel_cpu {

class Node;
class Edge;

using GlobalExecutionIndex = std::unordered_map<std::shared_ptr<Node>, std::pair<int, int>>;

struct AllocationContext {
    std::vector<std::shared_ptr<Edge>> edges;
    GlobalExecutionIndex execIndex;
    std::vector<size_t> syncPoints;
};

}  // namespace ov::intel_cpu
