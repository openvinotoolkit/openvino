// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/rank.hpp"
#include "ngraph/shape.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ngraph {
using ov::PartialShape;
}  // namespace ngraph
