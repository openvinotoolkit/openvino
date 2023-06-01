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

#include <limits>

#include "ngraph/op/util/max_pool_base.hpp"
#include "openvino/op/max_pool.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::MaxPool;
}  // namespace v1

namespace v8 {
using ov::op::v8::MaxPool;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
