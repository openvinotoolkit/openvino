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

#include <cstddef>
#include <ostream>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ngraph {
namespace op {
using ov::op::AutoBroadcastSpec;
using ov::op::AutoBroadcastType;
using ov::op::BroadcastModeSpec;
using ov::op::BroadcastType;
using ov::op::EpsMode;
using ov::op::PadMode;
using ov::op::PadType;
using ov::op::RecurrentSequenceDirection;
using ov::op::RoundingType;
using ov::op::TopKMode;
using ov::op::TopKSortType;
}  // namespace op
}  // namespace ngraph
