// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <openvino/op/util/roi_align_base.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace roi_align {
namespace validate {
/**
 * @brief Validates ROIs align input data and ROIs element type.
 *
 * @param op Pointer to ROIs align node.
 * @return Valid ROIs align output element type.
 */
inline element::Type data_and_roi_et(const Node* const op) {
    auto out_et = element::dynamic;

    const auto& input_et = op->get_input_element_type(0);
    const auto& rois_et = op->get_input_element_type(1);

    NODE_VALIDATION_CHECK(op,
                          element::Type::merge(out_et, input_et, rois_et) && out_et.is_real(),
                          "The data type for input and ROIs is expected to be a same floating point type. Got: ",
                          input_et,
                          " and: ",
                          rois_et);
    return out_et;
}

/**
 * @brief Check ROIs align batch indicies input element type.
 *
 * @param op  Pointer to ROIs align node.
 */
inline void batch_indicies_et(const Node* const op) {
    const auto& indicies_et = op->get_input_element_type(2);

    NODE_VALIDATION_CHECK(op,
                          indicies_et.is_integral_number(),
                          "The data type for batch indices is expected to be an integer. Got: ",
                          indicies_et);
}
}  // namespace validate

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ROIAlignBase* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    using TDim = typename TShape::value_type;

    const auto& input_ps = input_shapes[0];
    const auto& rois_ps = input_shapes[1];
    const auto& batch_indices_ps = input_shapes[2];

    const auto rois_ps_rank = rois_ps.rank();
    const auto input_ps_rank = input_ps.rank();
    const auto batch_indices_ps_rank = batch_indices_ps.rank();

    auto output_shapes = std::vector<TRShape>(1);
    auto& out_shape = output_shapes.front();
    out_shape.reserve(4);

    NODE_VALIDATION_CHECK(op, input_ps_rank.compatible(4), "Expected a 4D tensor for the input data. Got: ", input_ps);
    NODE_VALIDATION_CHECK(op, rois_ps_rank.compatible(2), "Expected a 2D tensor for the ROIs input. Got: ", rois_ps);
    NODE_VALIDATION_CHECK(op,
                          batch_indices_ps_rank.compatible(1),
                          "Expected a 1D tensor for the batch indices input. Got: ",
                          batch_indices_ps);

    if (rois_ps_rank.is_static()) {
        const auto& rois_second_dim = rois_ps[1];
        NODE_VALIDATION_CHECK(op,
                              rois_second_dim.compatible(op->get_rois_input_second_dim_size()),
                              "The second dimension of ROIs input should contain box coordinates. "
                              "op dimension is expected to be equal to ",
                              op->get_rois_input_second_dim_size(),
                              ". Got: ",
                              rois_second_dim);

        out_shape.push_back(rois_ps[0]);
    } else {
        out_shape.push_back(Dimension::dynamic());
    }

    NODE_VALIDATION_CHECK(
        op,
        batch_indices_ps_rank.is_dynamic() || TDim::merge(out_shape[0], batch_indices_ps[0], out_shape[0]),
        "The first dimension of ROIs input must be equal to the first dimension of the batch indices input. Got: ",
        out_shape[0],
        " and: ",
        batch_indices_ps[0]);

    out_shape.push_back(input_ps_rank.is_static() ? input_ps[1] : Dimension::dynamic());
    out_shape.emplace_back(static_cast<typename TDim::value_type>(op->get_pooled_h()));
    out_shape.emplace_back(static_cast<typename TDim::value_type>(op->get_pooled_w()));

    return output_shapes;
}
}  // namespace roi_align
}  // namespace op
}  // namespace ov
