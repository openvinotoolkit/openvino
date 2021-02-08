// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <cmath>
#include <set>
#include <sstream>
#include <utility>

#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/interp.hpp"
#include <transformations/rt_info/primitives_priority_attribute.hpp>
#include "legacy/ngraph_ops/scaleshift.hpp"

#include "exec_graph_info.hpp"

#include <cnn_network_ngraph_impl.hpp>
#include <precision_utils.h>
#include <cpp/ie_cnn_network.h>
#include <ngraph/ngraph.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <legacy/convert_function_to_cnn_network.hpp>
#include "legacy/graph_transformer.h"
#include "legacy/graph_tools.hpp"
#include "legacy/net_pass.h"
#include <legacy/cnn_network_impl.hpp>
#include <ie_cnn_layer_builder_ngraph.h>

namespace InferenceEngine {
namespace Builder {

template <>
std::string asString<double>(const double& value) {
    std::ostringstream sStrm;
    sStrm.precision(std::numeric_limits<double>::digits10);
    sStrm << std::fixed << value;
    std::string result = sStrm.str();

    auto pos = result.find_last_not_of("0");
    if (pos != std::string::npos) result.erase(pos + 1);

    pos = result.find_last_not_of(".");
    if (pos != std::string::npos) result.erase(pos + 1);

    return result;
}
}  // namespace Builder
}  // namespace InferenceEngine
