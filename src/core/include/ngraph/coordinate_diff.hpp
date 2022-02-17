// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
