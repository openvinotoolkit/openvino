// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace ov {
class Model;
namespace intel_cpu {
class Graph;

std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph &graph);
#ifdef CPU_DEBUG_CAPS
void serialize(const Graph &graph);
void serialize(const Graph &graph, const std::string& path);
void summary_perf(const Graph &graph);
#endif // CPU_DEBUG_CAPS
void average_counters(const Graph &graph);

}   // namespace intel_cpu
}   // namespace ov
