// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations_visibility.hpp"

bool labels_eq_or_eq_static_dims(const ov::Dimension& lhs, const ov::Dimension& rhs);

bool last_two_dims_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs);

bool reshape_keeps_last_two_dims(const std::shared_ptr<ov::Node>& op);

bool batches_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs, bool one_dim_can_differ = false);

bool batches_are_equal(const std::shared_ptr<ov::Node>& op_0, const std::shared_ptr<ov::Node>& op_1);

namespace ov {
namespace symbol {
namespace util {
/// \brief Collects labels from shape. Labels of static dimensions are guaranteed to be ov::no_labels
///
/// \param shape    Shape object to collect labels from
/// \param labels   TensorLabel object to collect labels to
///
/// \return Status of collecting the labels (false if rank is static else true)
TRANSFORMATIONS_API bool get_labels(const ov::PartialShape& shape, ov::TensorLabel& labels);

/// \brief Collects labels from tensor of Output object
///
/// \param output   Output object to collect labels from
/// \param labels   TensorLabel object to collect labels to
///
/// \return Status of collecting the labels (false if tensor has no labels else true)
TRANSFORMATIONS_API bool get_labels(const ov::Output<ov::Node>& output, ov::TensorLabel& labels);

/// \brief Compares
///
/// \param lhs   TensorLabel object to compare
/// \param rhs   TensorLabel object to compare
///
/// \return true if labels are unique and equal between lhs and rhs else false
TRANSFORMATIONS_API bool are_unique_and_equal_labels(const ov::TensorLabel& lhs, const ov::TensorLabel& rhs);

/// \brief Compares dimensions: if dimensions are static compares values of dimensions, if dimensions are dynamic
/// compares their respective labels using TableOfEquivalence
///
/// \param lhs   Dimension object to compare
/// \param rhs   Dimension object to compare
///
/// \return true if static dimensions are equal and dynamic dimensions have equal labels else false
TRANSFORMATIONS_API bool dims_are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs);

}  // namespace util
}  // namespace symbol
}  // namespace ov
