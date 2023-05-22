// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/op/op.hpp"
#include "openvino/op/space_to_batch.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::SpaceToBatch;
}  // namespace v1
using v1::SpaceToBatch;
}  // namespace op
}  // namespace ngraph
