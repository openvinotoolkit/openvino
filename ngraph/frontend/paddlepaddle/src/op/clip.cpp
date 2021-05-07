// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "clip.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs clip (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto min = node.get_attribute<float>("min");
    auto max = node.get_attribute<float>("max");
    PDPD_ASSERT(max >= min, "clip: max value must greater than min value!");
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Clamp>(data, min, max)}, {"Out"});     
}

}}}}