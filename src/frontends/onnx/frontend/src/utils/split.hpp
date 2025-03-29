// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief      Split value on specified axis into multiple parts.
///
/// \param  value          The value to be split.
/// \param  split_lengths  The vector defining the lengths of each split part.
/// \param  axis           The axis we split input node on. Default value is zero
///                        axis.
/// \note       This implementation supports negative `axis` values (similar to NumPy
///             indexing). This means that the axis to split on will be counted from
///             the back of the tensor (negative values are subtracted from its rank).
///
/// \return     The vector containing multiple outputs we split input node into.
///             The vector is output of Split:v1 op
///
ov::OutputVector make_split(const Output<Node>& value, const std::vector<int64_t>& split_lengths, int64_t axis = 0);

/// \brief      Split value on specified axis into multiple parts.
///
/// \param  value         The value to split.
/// \param  num_splits    The number of parts we want to split output at given
///                       axis. The length of the axis to split must be divisible by
///                       this value.
/// \param  axis          The axis we split input node on. Default value is zero
///                       axis.
///
/// \note       This implementation supports negative `axis` values (similar to NumPy
///             indexing). This means that the axis to split on will be counted from
///             the back of the tensor (negative values are subtracted from its rank).
///
/// \return     The vector containing multiple nodes we split input node into.
///             The vector is output of VariadicSplit:v1 op
///
ov::OutputVector make_split(const Output<Node>& value, int64_t num_splits, int64_t axis = 0);
}  // namespace util
}  // namespace op
}  // namespace ov
