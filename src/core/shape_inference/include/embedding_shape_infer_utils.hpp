// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/node.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {
namespace embedding {
/**
 * \brief Return a copy of the `emb_table_shape` with the first dimension replaced
 *        by the first dimension from the `dim_shape_src`
 *
 * \tparam TShape          Shape type
 *
 * \param op               Pointer to operator.
 * \param emb_table_shape  The shape to be copied
 * \param dim_shape_src    The shape to copy the first dimension from, with dynamic or static rank > 1
 *
 * \return The copy of the `emb_table_shape` with the first dimsnsion overwritten by `dim_shape_src[0]` if the rank is
 * static, otherwise fully dynamic shape with dynamic rank.
 */
template <class TShape, class TRShape = result_shape_t<TShape>>
TRShape out_shape_infer(const ov::Node* op, const TShape& emb_table_shape, const TShape& dim_shape_src) {
    if (emb_table_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, emb_table_shape.size() > 0, "EMB_TABLE can't be a scalar.");
        auto out_shape = TRShape(emb_table_shape);
        out_shape[0] = dim_shape_src.rank().is_static() ? dim_shape_src[0] : Dimension::dynamic();
        return out_shape;
    }
    return PartialShape::dynamic();
}

}  // namespace embedding
}  // namespace util
}  // namespace op
}  // namespace ov
