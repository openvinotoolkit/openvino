// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/space_to_batch.hpp>
#include <openvino/opsets/opset2.hpp>

#include "utils.hpp"

template <class T>
void shape_infer(const ov::opset2::SpaceToBatch* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const ov::Rank data_rank = data_shape.rank();
    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              (data_shape.size() >= 2),
                              "The data tensor with rank lower than 2 is not supported (data rank: ",
                              data_shape.size(),
                              ")");
    }

    std::vector<int64_t> block_val, pads_begin_val, pads_end_val;

    if (get_data_as_int64<T>(1, op, block_val, constant_data) &&
        get_data_as_int64<T>(2, op, pads_begin_val, constant_data) &&
        get_data_as_int64<T>(3, op, pads_end_val, constant_data) && data_shape.is_static()) {
        const ov::Shape& data_sshape = data_shape.to_shape();

        int64_t block_prod = 1;
        for (long idx : block_val)
            block_prod *= idx;

        ov::Shape output_shape = {static_cast<size_t>(data_sshape[0] * block_prod)};
        for (size_t idx = 1; idx < data_sshape.size(); ++idx) {
            NODE_VALIDATION_CHECK(op, block_val.at(idx) > 0, "block_shape values must be greater than 0");
            NODE_VALIDATION_CHECK(
                op,
                (pads_begin_val.at(idx) + data_sshape.at(idx) + pads_end_val.at(idx)) % block_val.at(idx) == 0,
                "The dimension on position: ",
                idx,
                " equal to: ",
                pads_begin_val.at(idx) + data_sshape.at(idx) + pads_end_val.at(idx),
                " must be a multiple of block_values[i]: ",
                block_val.at(idx));
            output_shape.push_back(static_cast<size_t>(pads_begin_val[idx] + data_sshape[idx] + pads_end_val[idx]) /
                                   block_val[idx]);
        }

        output_shapes[0] = T{output_shape};
    } else {
        set_output_to_be_partial(data_rank, output_shapes[0]);
    }
}