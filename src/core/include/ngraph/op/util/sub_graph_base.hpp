// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
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
