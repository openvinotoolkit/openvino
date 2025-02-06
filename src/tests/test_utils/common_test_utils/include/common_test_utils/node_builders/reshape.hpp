// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
/// \brief      Change shape of a value
///
/// \param[in]  value  The value to be reshaped.
/// \param[in]  shape  The new shape.
///
/// \return     Reshape:v1 op.
std::shared_ptr<Node> reshape(const Output<Node>& value, const Shape& shape);
}  // namespace utils
}  // namespace test
}  // namespace ov
