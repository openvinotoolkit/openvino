// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/scatter_base.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/scatter_update.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ScatterUpdate;
}  // namespace v3
}  // namespace op
}  // namespace ngraph
