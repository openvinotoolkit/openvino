// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace util {

/// \brief Normalize value to the max if value is negative.
///
/// \param value  Input value to normalize.
/// \param max    Value used for normalization
///
/// \return Value if positive otherwise return value + max
OPENVINO_API int64_t normalize(const int64_t& value, const int64_t& max);

/// \brief Check if values in vector are unique.
///
/// \param data  Input data to check.
///
/// \return True if unique otherwise false.
OPENVINO_API bool are_unique(const std::vector<int64_t>& data);

/// \brief Clip value to minimum if below min, or to maximum of above max.
///
/// \param value  Value to be clipped.
/// \param min    Minimum value bound.
/// \param max    Maximum value bound.
///
/// \return Value if between min, max otherwise min or max.
OPENVINO_API int64_t clip(const int64_t& value, const int64_t& min, const int64_t& max);

/// \brief Constant folds a subgraph to a constant node
///
/// \param subgraph sink
///
/// \return Constant node or nullptr if unable to constant fold the subgraph
OPENVINO_API std::shared_ptr<op::v0::Constant> constantfold_subgraph(const Output<Node>& subgraph_sink);

/// \brief Runs an estimation of source tensor. If it succeeded to calculate both bounds and
/// they are the same returns Constant operation from the resulting bound, otherwise nullptr.
///
/// \param source  Node output used to get its tensor data as constant.
/// \return Shared pointer to constant data or nullptr.
OPENVINO_API std::shared_ptr<op::v0::Constant> get_constant_from_source(const Output<Node>& source);

/// \brief Make scalar tensor which stores maximum value of ov::element::Type.
/// \param et  Element type to get its maximum.
/// \return Tensor with maximum value.
Tensor make_tensor_of_max_value(const element::Type_t et);

/// \brief Make scalar tensor which stores minimum value of ov::element::Type.
/// \param et  Element type to get its minimum.
/// \return Tensor with minimum value.
Tensor make_tensor_of_min_value(const element::Type_t et);

/// \brief Apply auto padding to padding_above and padding_below inputs
///        if all needed informations are known.
///
/// \param image_shape       The shape of input image.
/// \param filter_shape      The shape of filter input.
/// \param filter_strides    The strides of applied padding.
/// \param filter_dilations  The dilations of applied padding.
/// \param pad_type          The type of padding. Auto padding is applied only
///                          for SAME_UPPER and SAME_LOWER mode.
/// \param padding_above     The beginning of padding shape.
/// \param end               The beginning of padding shape.
///
/// \return true if auto padding was applied successfully (all needed informations such as
///         spatial dims are known), false otherwise.
OPENVINO_API
bool try_apply_auto_padding(const PartialShape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below);

/// @brief Get the tensors shapes as ov::PartialShape.
///
/// @param tensors  Input tensors vector to get their shapes.
/// @return Vector of partial shapes same size as input tensor vector.
OPENVINO_API std::vector<PartialShape> get_tensors_partial_shapes(const TensorVector& tensors);
}  // namespace util
}  // namespace ov
