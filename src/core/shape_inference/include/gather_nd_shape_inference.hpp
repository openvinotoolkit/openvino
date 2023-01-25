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

    // check ranks of input tensors
    const auto& data_pshape = input_shapes[0];
    const auto& indices_pshape = input_shapes[1];

    if (data_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, data_pshape.rank().get_length() > 0, "Data rank must be at least 1.");

        NODE_VALIDATION_CHECK(op,
                              data_pshape.rank().get_length() > static_cast<int64_t>(op->get_batch_dims()),
                              "Number of batch dimensions must not exceed a rank of data.");
    }

    if (indices_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, indices_pshape.rank().get_length() > 0, "Indices rank must be at least 1.");

        NODE_VALIDATION_CHECK(op,
                              indices_pshape.rank().get_length() > static_cast<int64_t>(op->get_batch_dims()),
                              "Number of batch dimensions must not exceed a rank of indices.");
    }

    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static()) {
        // check that batch dimensions of data and indices are the same
        for (size_t batch_dim = 0; batch_dim < op->get_batch_dims(); batch_dim++) {
            if (data_pshape[batch_dim].is_static() && indices_pshape[batch_dim].is_static()) {
                NODE_VALIDATION_CHECK(op,
                                      data_pshape[batch_dim].get_length() == indices_pshape[batch_dim].get_length(),
                                      "Batch dimensions of data and indices must be the same.");
            }
        }

        if (indices_pshape[indices_pshape.rank().get_length() - 1].is_static()) {
            NODE_VALIDATION_CHECK(
                op,
                static_cast<int64_t>(indices_pshape[indices_pshape.rank().get_length() - 1].get_length() +
                                     op->get_batch_dims()) <= data_pshape.rank().get_length(),
                "Length of a tuple with indices must not exceed a rank of data tensor "
                "excluding "
                "batch dimensions.");
        }
    }

    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static() &&
        indices_pshape[indices_pshape.rank().get_length() - 1].is_static()) {
        auto batch_dims = op->get_batch_dims();
        auto indices_tuple_length = indices_pshape[indices_pshape.rank().get_length() - 1].get_length();
        int64_t slice_length = data_pshape.rank().get_length() - indices_tuple_length - batch_dims;
        int64_t output_indices_length = indices_pshape.rank().get_length() - batch_dims - 1;
        auto output_rank = output_indices_length + slice_length;
        size_t delta_output_rank = 0;
        delta_output_rank = batch_dims;
        std::vector<Dimension> output_shape(output_rank + delta_output_rank);
        for (size_t dim = 0; dim < batch_dims; dim++) {
            output_shape[dim] = 1;
            if (data_pshape[dim].is_static()) {
                output_shape[dim] = data_pshape[dim].get_length();
            } else if (indices_pshape[dim].is_static()) {
                output_shape[dim] = indices_pshape[dim].get_length();
            } else {
                output_shape[dim] = Dimension::dynamic();
                break;
            }
        }
        for (int64_t dim = 0; dim < output_indices_length; dim++) {
            output_shape[dim + delta_output_rank] = indices_pshape[dim + batch_dims];
        }
        for (int64_t dim = 0; dim < slice_length; dim++) {
            output_shape[output_indices_length + dim + delta_output_rank] =
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

    // If we have m_batch_dims > 1 we need to fuse batch dimensions of output
    auto m_batch_dims = op->get_batch_dims();
    if (m_batch_dims > 1 && output_shapes[0].rank().is_static()) {
        const auto& output_pshape = output_shapes[0];
        const auto& out_size = output_pshape.size();
        std::vector<DimType> output_shape(out_size - m_batch_dims + 1);
        output_shape[0] = 1;
        for (size_t dim = 0; dim < m_batch_dims; dim++) {
            if (output_pshape[dim].is_static()) {
                output_shape[0] *= output_pshape[dim].get_length();
            } else {
                output_shape[0] = Dimension::dynamic();
                break;
            }
        }
        size_t ind = 1;
        for (size_t dim = m_batch_dims; dim < out_size; dim++) {
            if (output_pshape[dim].is_static()) {
                output_shape[ind] = output_pshape[dim].get_length();
            } else {
                output_shape[ind] = Dimension::dynamic();
            }
            ind++;
        }
        // Update output shape
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
