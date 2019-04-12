//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "ie_icnn_network.hpp"
#include "mkldnn_graph.h"

#include <memory>

namespace MKLDNNPlugin {

    void dump_graph_as_dot(const MKLDNNGraph &graph, std::ostream &out);

    std::shared_ptr<InferenceEngine::ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph);

}  // namespace MKLDNNPlugin
