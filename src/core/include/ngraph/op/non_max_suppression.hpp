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

#include "ngraph/op/op.hpp"
#include "openvino/op/non_max_suppression.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::NonMaxSuppression;
}  // namespace v1

namespace v3 {
using ov::op::v3::NonMaxSuppression;
}  // namespace v3

namespace v4 {
using ov::op::v4::NonMaxSuppression;
}  // namespace v4

namespace v5 {
using ov::op::v5::NonMaxSuppression;
}  // namespace v5

namespace v9 {
using ov::op::v9::NonMaxSuppression;
}  // namespace v9
}  // namespace op
using ov::operator<<;
}  // namespace ngraph
