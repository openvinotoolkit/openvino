// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
namespace embedding {
/**
 * \brief Return a copy of the source shape with the first dimension replaced by provided dim
 *
 * \tparam TShape        Shape type
 * \tparam TDim          Dimension type
 *
 * \param op             Pointer to operator.
 * \param shape_src      The shape to be copied
 * \param dim            The replacement dimension to overwirite the first one
 *
 * \return The copy of the `shape_src` with `dim` as the first element if the rank is static,
 *         otherwise fully dynamic shape with dynamic rank
 */
template <class TShape, class TDim = typename TShape::value_type>
TShape copy_shape_and_update_dim(const ov::Node* op, const TShape& shape_src, const TDim& dim) {
    if (shape_src.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, shape_src.size() > 0, "EMB_TABLE can't be a scalar.");
        auto shape_copy = shape_src;
        shape_copy[0] = dim;
        return shape_copy;
    }
    return PartialShape::dynamic();
}

template <class TShape>
TShape copy_shape_and_update_dim(const ov::Node* op, const TShape& shape_src, const TShape& dim_src) {
    return copy_shape_and_update_dim(op, shape_src, dim_src.rank().is_static() ? dim_src[0] : Dimension::dynamic());
}

}  // namespace embedding
}  // namespace util
}  // namespace op
}  // namespace ov
