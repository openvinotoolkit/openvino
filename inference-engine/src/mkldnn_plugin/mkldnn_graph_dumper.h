// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_icnn_network.hpp"
#include "mkldnn_graph.h"

#include <memory>

namespace MKLDNNPlugin {

    void dump_graph_as_dot(const MKLDNNGraph &graph, std::ostream &out);

    std::shared_ptr<InferenceEngine::ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph);

}  // namespace MKLDNNPlugin
