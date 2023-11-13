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

#include "ngraph/op/util/gather_base.hpp"
#include "openvino/op/gather.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Gather;
}  // namespace v1
namespace v7 {
using ov::op::v7::Gather;
}  // namespace v7
namespace v8 {
using ov::op::v8::Gather;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
