// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/roll.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v7 {

template <class T>
void shape_infer(const ov::op::v7::Roll* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
#if 0
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    const auto& class_probs_ps = input_shapes[0];
    const auto& bbox_deltas_ps = input_shapes[1];
    const auto& image_shape_ps = input_shapes[2];

    NODE_VALIDATION_CHECK(op,
                          class_probs_ps.rank().compatible(4),
                          "Proposal layer shape class_probs should be rank 4 compatible (",
                          class_probs_ps,
                          ").");

    NODE_VALIDATION_CHECK(op,
                          bbox_deltas_ps.rank().compatible(4),
                          "Proposal layer shape bbox_deltas should be rank 4 compatible (",
                          bbox_deltas_ps,
                          ").");

    NODE_VALIDATION_CHECK(op,
                          image_shape_ps.rank().compatible(1),
                          "Proposal layer shape image_shape should be rank 1 compatible (",
                          image_shape_ps,
                          ").");
    if (bbox_deltas_ps.is_static() && class_probs_ps.is_static()) {
        // class probs and bbox deltas shapes are static, check anchor count and batch number
        // consistency
        NODE_VALIDATION_CHECK(op,
                              class_probs_ps[1].get_length() * 2 == bbox_deltas_ps[1].get_length(),
                              "Anchor number inconsistent between class_probs (",
                              class_probs_ps[1].get_length() / 2,
                              "), and bbox_deltas (",
                              bbox_deltas_ps[1].get_length() / 4,
                              ").");

        NODE_VALIDATION_CHECK(op,
                              class_probs_ps[0] == bbox_deltas_ps[0],
                              "Batch size inconsistent between class_probs (",
                              class_probs_ps[0],
                              ") and bbox deltas (",
                              bbox_deltas_ps[0],
                              ").");
    }

    if (image_shape_ps.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              image_shape_ps[0].get_length() >= 3 && image_shape_ps[0].get_length() <= 4,
                              "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
                              image_shape_ps[0],
                              ").");
    }

    auto out_dim = DimType{};

    if (class_probs_ps.rank().is_static() && bbox_deltas_ps.rank().is_static()) {
        out_dim = (class_probs_ps[0] & bbox_deltas_ps[0]);
    } else if (class_probs_ps.rank().is_static()) {
        out_dim = class_probs_ps[0];
    } else if (bbox_deltas_ps.rank().is_static()) {
        out_dim = bbox_deltas_ps[0];
    } else {
        proposal_build_dynamic_dimension(out_dim);
    }
    output_shapes[0].resize(2);
    (output_shapes[0])[0] = out_dim * op->get_attrs().post_nms_topn;
    (output_shapes[0])[1] = 5;
#endif
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);

    const auto& data_pshape = input_shapes[0];
    const auto& shift_pshape = input_shapes[1];
    const auto& axes_pshape = input_shapes[2];

    if (shift_pshape.rank().is_static()) {
        const auto& shift_rank = shift_pshape.size();
        NODE_VALIDATION_CHECK(this, shift_rank <= 1, "Shift must be a scalar or 1D tensor.");
    }

    if (axes_pshape.rank().is_static() {
        const auto& axes_rank = axes_pshape.size();
        NODE_VALIDATION_CHECK(this, axes_rank <= 1, "Axes must be a scalar or 1D tensor.");
    }

    // If shift is a scalar, than axes can be arbitrary 1d tensor and we don't need
    // to check shift shape consistency with axes, otherwise the check is needed.
    if (!(shift_pshape.is_static() && ngraph::is_scalar(shift_pshape.to_shape()))) {
        NODE_VALIDATION_CHECK(this,
                              shift_pshape.compatible(axes_pshape),
                              "If shift is a 1D vector, axes must be a 1D tensor of the same size.");
    }
    std::vector<int64_t> axes{};

    if (get_data_as_int64<T>(2, op, axes, constant_data)) {
        if (data_pshape.rank().is_static()) {
            const auto& data_rank = data_pshape.size();
            for (int64_t& axis : axes) {
                NODE_VALIDATION_CHECK(this,
                                      axis < data_rank,
                                      "Axes must be less than data tensor rank. Got "
                                      "data tensor rank: ",
                                      data_rank,
                                      ", axis: ",
                                      axis);
                if (axis < 0) {
                    axis += data_rank;
                }
                NODE_VALIDATION_CHECK(this,
                                      axis >= 0,
                                      "Axes must be positive or equal to zero. Got "
                                      "axis: ",
                                      axis);
            }
        }
    }

    output_shapes[0] = input_shapes[0];
}

}  // namespace v7
}  // namespace op
}  // namespace ov