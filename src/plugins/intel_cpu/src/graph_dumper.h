// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "graph.h"
#include "utils/debug_capabilities.h"

#include <memory>

namespace ov {
namespace intel_cpu {

std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph &graph);
#ifdef CPU_DEBUG_CAPS
void serialize(const Graph &graph);
void summary_perf(const Graph &graph);
#endif // CPU_DEBUG_CAPS

}   // namespace intel_cpu
}   // namespace ov
