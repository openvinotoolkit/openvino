// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include <cstddef>
#include <ostream>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "openvino/core/coordinate_diff.hpp"

namespace ngraph {
/// \brief A difference (signed) of tensor element coordinates.
using ov::CoordinateDiff;
}  // namespace ngraph
