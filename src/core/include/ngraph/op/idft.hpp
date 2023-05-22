// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include <cstddef>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fft_base.hpp"
#include "openvino/op/idft.hpp"

namespace ngraph {
namespace op {
namespace v7 {
using ov::op::v7::IDFT;
}  // namespace v7
}  // namespace op
}  // namespace ngraph
