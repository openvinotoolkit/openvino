// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/deformable_psroi_pooling.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DeformablePSROIPooling* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 || input_shapes.size() == 3);

    const auto& input_pshape = input_shapes[0];
    const auto& box_coords_pshape = input_shapes[1];

    NODE_VALIDATION_CHECK(op,
                          input_pshape.rank().compatible(4),
                          "First input rank must be compatible with 4 (input rank: ",
                          input_pshape.rank(),
                          ")");
    NODE_VALIDATION_CHECK(op,
                          box_coords_pshape.rank().compatible(2),
                          "Second input rank must be compatible with 2 (input rank: ",
                          box_coords_pshape.rank(),
                          ")");

    if (input_shapes.size() == 3)  // offsets input is provided
    {
        const auto& offsets_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              offsets_shape.rank().compatible(4),
                              "Third input rank must be compatible with 4 (input rank: ",
                              offsets_shape.rank(),
                              ")");
    }

    NODE_VALIDATION_CHECK(op, op->get_output_dim() > 0, "Value of `output_dim` attribute has to be greater than 0 ");
    NODE_VALIDATION_CHECK(op, op->get_group_size() > 0, "Value of `group_size` attribute has to be greater than 0 ");

    using DimType = typename TShape::value_type;
    using DimTypeVal = typename DimType::value_type;
    // The output shape: [num_rois, output_dim, group_size, group_size]
    return {TRShape{box_coords_pshape.rank().is_static() ? box_coords_pshape[0] : DimType{},
                    static_cast<DimTypeVal>(op->get_output_dim()),
                    static_cast<DimTypeVal>(op->get_group_size()),
                    static_cast<DimTypeVal>(op->get_group_size())}};
}
}  // namespace v1
}  // namespace op
}  // namespace ov
