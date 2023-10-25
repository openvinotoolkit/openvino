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

#include <cstdint>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/interpolate.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using InterpolateAttrs = ov::op::v0::Interpolate::Attributes;
using ov::op::v0::Interpolate;
}  // namespace v0
namespace v4 {
using ov::op::v4::Interpolate;
}  // namespace v4
namespace v11 {
using ov::op::v11::Interpolate;
}  // namespace v11
using v0::Interpolate;
using v0::InterpolateAttrs;
}  // namespace op
}  // namespace ngraph
