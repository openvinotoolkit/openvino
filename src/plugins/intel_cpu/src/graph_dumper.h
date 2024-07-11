// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"

#include <memory>

namespace ov {
namespace intel_cpu {

std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph &graph);
#ifdef CPU_DEBUG_CAPS
void serialize(const Graph &graph);
void summary_perf(const Graph &graph);
void average_counters(const Graph &graph);
#endif // CPU_DEBUG_CAPS

}   // namespace intel_cpu
}   // namespace ov
