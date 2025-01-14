// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/gather_nd.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace gather_nd {
template <class TOp, class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> gather_nd_base_shape_infer(const TOp* op, const std::vector<TShape>& input_shapes) {
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
            cmp::le(indices_tuple_length + op->get_batch_dims(), data_pshape.rank().get_length()),
            "Length of a tuple with indices must not exceed a rank of data tensor excluding batch dimensions.");
        const auto slice_length = data_pshape.size() - indices_tuple_length - batch_dims;
        const auto output_indices_length = indices_pshape.size() - batch_dims - 1;

        using DimType = typename TShape::value_type;
        std::vector<DimType> output_dims(batch_dims);
        output_dims.reserve(batch_dims + output_indices_length + slice_length);
        // Merge batch dimensions
        for (size_t dim_idx = 0; dim_idx < batch_dims; ++dim_idx) {
            NODE_VALIDATION_CHECK(op,
                                  DimType::merge(output_dims[dim_idx], data_pshape[dim_idx], indices_pshape[dim_idx]),
                                  "Batch dimensions of data and indices must be the same.");
        }
        // Insert middle dimensions from the indices shape
        for (auto dim_idx = batch_dims; dim_idx < indices_pshape.size() - 1; ++dim_idx) {
            output_dims.emplace_back(indices_pshape[dim_idx]);
        }
        // Insert dimensions fully taken from the data shape
        for (auto dim_idx = batch_dims + indices_tuple_length; dim_idx < data_pshape.size(); ++dim_idx) {
            output_dims.emplace_back(data_pshape[dim_idx]);
        }
        return {TRShape(std::move(output_dims))};
    } else {
        return {ov::PartialShape::dynamic()};
    }
}
}  // namespace gather_nd
namespace v5 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GatherND* op, const std::vector<TShape>& input_shapes) {
    using DimType = typename TShape::value_type;
    auto output_shapes = gather_nd::gather_nd_base_shape_infer(op, input_shapes);

    // If batch_dims > 1, batch dimensions are need to be fused
    auto batch_dims = op->get_batch_dims();
    if (batch_dims > 1 && output_shapes[0].rank().is_static()) {
        auto& output_base_shape = output_shapes[0];
        auto output_dims = std::vector<DimType>{output_base_shape[0]};
        std::for_each(output_base_shape.begin() + 1,
                      output_base_shape.begin() + batch_dims,
                      [&output_dims](const DimType& dim) {
                          output_dims[0] *= dim;
                      });
        output_dims.insert(output_dims.begin() + 1, output_base_shape.begin() + batch_dims, output_base_shape.end());
        output_shapes[0] = TRShape(std::move(output_dims));
    }
    return output_shapes;
}
}  // namespace v5

namespace v8 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GatherND* op, const std::vector<TShape>& input_shapes) {
    return gather_nd::gather_nd_base_shape_infer(op, input_shapes);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
