// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "openvino/op/tensor_iterator.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::TensorIterator;
}  // namespace v0
using v0::TensorIterator;
}  // namespace op
}  // namespace ngraph
