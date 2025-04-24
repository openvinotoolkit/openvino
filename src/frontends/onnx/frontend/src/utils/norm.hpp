// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief      Creates node which calculates L-p norm on input tensor.
///
/// \param[in]  value           The input tensor.
/// \param[in]  reduction_axes  The axes along which we calculate norm.
/// \param[in]  p_norm          The p norm to calculate.
/// \param[in]  bias            The bias added to the calculated sum.
/// \param[in]  keep_dims       The flag indicates if axes will be removed or kept.
///
/// \return     L-p norm of value. The output sub-graph is composed of v1 ops.
///
std::shared_ptr<Node> lp_norm(const Output<Node>& value,
                              const Output<Node>& reduction_axes,
                              std::size_t p_norm = 2,
                              float bias = 0.f,
                              bool keep_dims = false);
}  // namespace util
}  // namespace op
}  // namespace ov
