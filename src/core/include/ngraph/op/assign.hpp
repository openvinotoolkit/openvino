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

#include "ngraph/op/sink.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_extension.hpp"
#include "openvino/op/assign.hpp"

namespace ngraph {
namespace op {
using ov::op::util::AssignBase;

namespace v3 {
using ov::op::v3::Assign;
}  // namespace v3
namespace v6 {
using ov::op::v6::Assign;
}  // namespace v6
}  // namespace op
}  // namespace ngraph
