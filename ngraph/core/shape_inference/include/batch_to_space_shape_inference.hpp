// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/batch_to_space.hpp>
#include <openvino/opsets/opset1.hpp>

namespace ov {
namespace op {
namespace v1 {

template <class T>
void set_output_partial(const Rank& rank, T& output_shape) {
    OPENVINO_UNREACHABLE("Shape Infer can't set partial shape");
}

template <>
void set_output_partial<ov::PartialShape>(const Rank& rank, ov::PartialShape& output_shape) {
    output_shape = ov::PartialShape::dynamic(rank);
}

template <class T>
bool get_data_as_int64(size_t idx,
                       const ov::Node* op,
                       std::vector<int64_t>& data_vec,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        data_vec = ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>();
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        data_vec = constant->cast_vector<int64_t>();
    }
    return true;
}

template <>
bool get_data_as_int64<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    std::vector<int64_t>& data_vec,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        data_vec = ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>();
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        data_vec = constant->cast_vector<int64_t>();
    } else {
        return false;
    }
    return true;
}

template <class T>
void shape_infer(const BatchToSpace* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];
    const ov::PartialShape& block_shape = input_shapes[1];
    const ov::PartialShape& crops_begin_shape = input_shapes[2];
    const ov::PartialShape& crops_end_shape = input_shapes[3];

    auto inputs_same_ps = crops_begin_shape;
    NODE_VALIDATION_CHECK(op,
                          T::merge_into(inputs_same_ps, crops_end_shape) && T::merge_into(inputs_same_ps, block_shape),
                          "block_shape, crops_begin and crops_end inputs must have the same shape. Got: ",
                          block_shape,
                          ", ",
                          crops_begin_shape,
                          " and ",
                          crops_end_shape);

    const Rank inputs_rank_one = inputs_same_ps.rank();
    NODE_VALIDATION_CHECK(op,
                          inputs_rank_one.compatible(1),
                          "block_shape and crops inputs must have rank 1. Got: ",
                          inputs_rank_one);

    const Rank data_rank = data_shape.rank();
    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              (data_rank.get_length() >= 2),
                              "data input must have rank greater or equal than 2. Got: ",
                              data_rank.get_length());

        if (inputs_same_ps.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  data_rank.get_length() == inputs_same_ps[0].get_length(),
                                  "block_shape and crop inputs must have same number of elements "
                                  "as data input rank. Got: ",
                                  inputs_same_ps[0],
                                  " and ",
                                  data_rank);
        }
    }

    std::vector<int64_t> block_val, crops_begin_val, crops_end_val;

    if (get_data_as_int64<T>(1, op, block_val, constant_data) &&
        get_data_as_int64<T>(2, op, crops_begin_val, constant_data) &&
        get_data_as_int64<T>(3, op, crops_end_val, constant_data) && data_shape.is_static()) {
        const ov::Shape& data_sshape = data_shape.to_shape();

        bool block_vals_valid = std::all_of(begin(block_val), end(block_val), [](int64_t elem) {
            return elem >= 1;
        });
        NODE_VALIDATION_CHECK(op, block_vals_valid, "Elements of block_shape input must be greater or equal to one.");

        bool crops_begin_vals_valid = std::all_of(begin(crops_begin_val), end(crops_begin_val), [](int64_t elem) {
            return elem >= 0;
        });
        bool crops_end_vals_valid = std::all_of(begin(crops_end_val), end(crops_end_val), [](int64_t elem) {
            return elem >= 0;
        });
        NODE_VALIDATION_CHECK(op,
                              crops_begin_vals_valid && crops_end_vals_valid,
                              "Elements of crops_begin and crops_end inputs must be greater or equal to zero.");

        int64_t block_prod = std::accumulate(begin(block_val), end(block_val), 1, std::multiplies<int64_t>());

        NODE_VALIDATION_CHECK(op,
                              data_sshape[0] % block_prod == 0,
                              "The input data's 'batch' axis size: ",
                              data_sshape[0],
                              " must be a multiple of",
                              " product of block_shape values: ",
                              block_prod);

        for (size_t idx = 0; idx < data_sshape.size(); idx++) {
            const bool is_valid_crops_and_shape =
                crops_begin_val[idx] + crops_end_val[idx] <= block_val[idx] * static_cast<int64_t>(data_sshape[idx]);
            NODE_VALIDATION_CHECK(op,
                                  is_valid_crops_and_shape,
                                  "crops_begin[i] + crops_end[i] must be less or equal to "
                                  "block_shape[i] * input_shape[i]");
        }

        ov::Shape output_sshape = {static_cast<size_t>(data_sshape[0] / block_prod)};
        for (size_t idx = 1; idx < data_sshape.size(); ++idx) {
            output_sshape.push_back(
                static_cast<size_t>(data_sshape[idx] * block_val[idx] - crops_begin_val[idx] - crops_end_val[idx]));
        }

        output_shapes[0] = T{output_sshape};
    } else {
        set_output_partial(data_rank, output_shapes[0]);
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov
