// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/ctc_loss.hpp>

namespace ov {
namespace op {
namespace v4 {

template <class T>
inline void dynamic_inference(T& output_shape) {
    OPENVINO_UNREACHABLE("This code should be executed only for PartialShape class");
}

template <>
inline void dynamic_inference<ov::PartialShape>(ov::PartialShape& output_shape) {
    output_shape = ov::PartialShape{Dimension::dynamic()};
}

template <class T>
void shape_infer(const CTCLoss* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 4 || input_shapes.size() == 5) && output_shapes.size() == 1);

    // check ranks of input tensors
    const auto& logits_pshape = input_shapes[0];
    const auto& logit_length_pshape = input_shapes[1];
    const auto& labels_pshape = input_shapes[2];
    const auto& label_length_pshape = input_shapes[3];

    NODE_VALIDATION_CHECK(op,
                          logits_pshape.rank().compatible(3),
                          "Expected a 3D tensor for logits. Got: ",
                          logits_pshape);

    NODE_VALIDATION_CHECK(op,
                          logit_length_pshape.rank().compatible(1),
                          "Expected a 1D tensor for logit length. Got: ",
                          logit_length_pshape);

    NODE_VALIDATION_CHECK(op,
                          labels_pshape.rank().compatible(2),
                          "Expected a 2D tensor for labels. Got: ",
                          labels_pshape);

    NODE_VALIDATION_CHECK(op,
                          label_length_pshape.rank().compatible(1),
                          "Expected a 1D tensor for label length. Got: ",
                          label_length_pshape);

    // check optional input shape: blank index
    if (input_shapes.size() == 5) {
        const auto& blank_index_pshape = input_shapes[4];
        NODE_VALIDATION_CHECK(op,
                              blank_index_pshape.rank().compatible(0),
                              "Expected a scalar for blank index. Got: ",
                              blank_index_pshape);
    }

    // check shapes of input tensors
    size_t batch_size = 1;
    bool is_batch_size_set = false;
    size_t time_steps = 1;
    bool is_time_steps_set = false;

    if (logits_pshape.rank().is_static()) {
        if (logits_pshape[0].is_static()) {
            batch_size = logits_pshape[0].get_length();
            is_batch_size_set = true;
        }
        if (logits_pshape[1].is_static()) {
            time_steps = logits_pshape[1].get_length();
            is_time_steps_set = true;
        }
    }

    if (logit_length_pshape.is_static()) {
        if (is_batch_size_set) {
            NODE_VALIDATION_CHECK(op,
                                  logit_length_pshape[0].compatible(batch_size),
                                  "The first dimension of logit length must be equal to the first dimension ",
                                  "of the logits. Got: ",
                                  logit_length_pshape[0],
                                  " and: ",
                                  batch_size);
        } else {
            batch_size = logit_length_pshape[0].get_length();
            is_batch_size_set = true;
        }
    }

    if (labels_pshape.rank().is_static()) {
        if (is_batch_size_set) {
            NODE_VALIDATION_CHECK(op,
                                  labels_pshape[0].compatible(batch_size),
                                  "The first dimension of labels must be equal to the first dimension ",
                                  "of the logits and the logit length. Got: ",
                                  labels_pshape[0],
                                  " and: ",
                                  batch_size);
        } else if (labels_pshape[0].is_static()) {
            batch_size = labels_pshape[0].get_length();
            is_batch_size_set = true;
        }

        if (is_time_steps_set) {
            NODE_VALIDATION_CHECK(op,
                                  labels_pshape[1].compatible(time_steps),
                                  "The second dimension of labels must be equal to the second dimension ",
                                  "of logits. Got: ",
                                  labels_pshape[1],
                                  " and: ",
                                  time_steps);
        }
    }

    if (label_length_pshape.is_static()) {
        if (!is_batch_size_set && label_length_pshape[0].is_static()) {
            batch_size = label_length_pshape[0].get_length();
            is_batch_size_set = true;
        }
        NODE_VALIDATION_CHECK(op,
                              label_length_pshape[0].compatible(batch_size),
                              "The first dimension of label length must be equal to the first dimension ",
                              "of the logits, the logit length and labels. Got: ",
                              label_length_pshape[0],
                              " and: ",
                              batch_size);
    }

    auto& output_shape = output_shapes[0];
    output_shape.resize(1);

    if (is_batch_size_set) {
        output_shape[0] = batch_size;
    } else {
        dynamic_inference(output_shape);
    }
}

}  // namespace v4
}  // namespace op
}  // namespace ov
