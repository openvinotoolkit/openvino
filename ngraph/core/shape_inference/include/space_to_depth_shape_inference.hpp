// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/space_to_depth.hpp>
#include <openvino/opsets/opset1.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const ov::op::v0::SpaceToDepth* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const ov::Rank data_rank = data_shape.rank();
    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              !(data_shape.size() < 3),
                              "The input tensor with rank lower than 3 is not supported (input rank: ",
                              data_shape.size(),
                              ")");
    }

    if (data_shape.is_static()) {
        const ov::Shape& data_sshape = data_shape.to_shape();
        auto multiplier = std::pow(op->m_blocksize, data_sshape.size() - 2);

        auto out_shape = data_sshape;
        out_shape[1] *= multiplier;
        for (size_t i = 2; i < out_shape.size(); i++) {
            NODE_VALIDATION_CHECK(op,
                                  op->m_blocksize > 0 && !(out_shape[i] % op->m_blocksize),
                                  "The dimension on position: ",
                                  i,
                                  " equal to: ",
                                  out_shape[i],
                                  " must be a multiple of m_blocksize: ",
                                  op->m_blocksize);

            out_shape[i] /= op->m_blocksize;
        }
        output_shapes[0] = T{out_shape};
    } else {
        set_output_to_be_partial(data_rank, output_shapes[0]);
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov