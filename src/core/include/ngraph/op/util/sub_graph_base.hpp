// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include <ngraph/op/parameter.hpp>

#include "ngraph/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::SubGraphOp;
using InputDescriptionPtr = util::SubGraphOp::InputDescription::Ptr;
using OutputDescriptionPtr = util::SubGraphOp::OutputDescription::Ptr;
using InputDescriptionVector = std::vector<InputDescriptionPtr>;
using OutputDescriptionVector = std::vector<OutputDescriptionPtr>;
}  // namespace util
}  // namespace op

}  // namespace ngraph
