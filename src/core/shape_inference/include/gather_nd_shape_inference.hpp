// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/gather_nd.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace gather_nd {
template <class TShape, class TOp>
std::vector<TShape> gather_nd_base_shape_infer(const TOp* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& data_pshape = input_shapes[0];
    const auto& indices_pshape = input_shapes[1];

    if (data_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, data_pshape.size() > 0, "Data rank must be at least 1.");
        NODE_VALIDATION_CHECK(op,
                              data_pshape.size() > op->get_batch_dims(),
                              "Number of batch dimensions must not exceed a rank of data.");
    }

    if (indices_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, indices_pshape.size() > 0, "Indices rank must be at least 1.");
        NODE_VALIDATION_CHECK(op,
                              indices_pshape.size() > op->get_batch_dims(),
                              "Number of batch dimensions must not exceed a rank of indices.");
    }

    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static() &&
        indices_pshape[indices_pshape.size() - 1].is_static()) {
        auto batch_dims = op->get_batch_dims();
        auto indices_tuple_length = indices_pshape[indices_pshape.size() - 1].get_length();  // last dim of indices

        NODE_VALIDATION_CHECK(
            op,
            static_cast<int64_t>(indices_tuple_length + op->get_batch_dims()) <= data_pshape.rank().get_length(),
            "Length of a tuple with indices must not exceed a rank of data tensor excluding batch dimensions.");
        int64_t slice_length = data_pshape.rank().get_length() - indices_tuple_length - batch_dims;
        int64_t output_indices_length = indices_pshape.rank().get_length() - batch_dims - 1;
        auto output_rank = output_indices_length + slice_length;

        using DimType = typename TShape::value_type;
        std::vector<DimType> output_shape(output_rank + batch_dims);
        for (size_t dim = 0; dim < batch_dims; ++dim) {
            NODE_VALIDATION_CHECK(op,
                                  DimType::merge(output_shape[dim], data_pshape[dim], indices_pshape[dim]),
                                  "Batch dimensions of data and indices must be the same.");
        }
        for (int64_t dim = 0; dim < output_indices_length; ++dim) {
            output_shape[batch_dims + dim] = indices_pshape[batch_dims + dim];
        }
        for (int64_t dim = 0; dim < slice_length; ++dim) {
            output_shape[batch_dims + output_indices_length + dim] =
                data_pshape[batch_dims + indices_tuple_length + dim];
        }
        return std::vector<TShape>{TShape(output_shape)};
    } else {
        return std::vector<TShape>{ov::PartialShape::dynamic()};
    }
}
}  // namespace gather_nd
namespace v5 {
template <class TShape>
void shape_infer(const GatherND* op, const std::vector<TShape>& input_shapes, std::vector<TShape>& output_shapes) {
    using DimType = typename TShape::value_type;
    output_shapes = gather_nd::gather_nd_base_shape_infer(op, input_shapes);

    // If batch_dims > 1, batch dimensions are need to be fused
    auto batch_dims = op->get_batch_dims();
    if (batch_dims > 1 && output_shapes[0].rank().is_static()) {
        const auto& output_base_shape = output_shapes[0];
        std::vector<DimType> output_shape{1};
        for (size_t dim = 0; dim < batch_dims; ++dim) {
            output_shape[0] *= output_base_shape[dim];
        }
        output_shape.insert(output_shape.begin() + 1, output_base_shape.begin() + batch_dims, output_base_shape.end());
        output_shapes[0] = TShape(output_shape);
    }
}
}  // namespace v5

namespace v8 {
template <class TShape>
void shape_infer(const GatherND* op, const std::vector<TShape>& input_shapes, std::vector<TShape>& output_shapes) {
    output_shapes = gather_nd::gather_nd_base_shape_infer(op, input_shapes);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
