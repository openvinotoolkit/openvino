// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/rnn_cell.hpp>

namespace ov {
namespace op {
namespace util {
template <class T>
void validate_input_rank(RNNCellBase* op, const std::vector<T>& input) {
    enum { X, initial_hidden_state, W, R, B };

    // Verify static ranks for all inputs
    for (size_t i = 0; i < input.size(); i++) {
        NODE_VALIDATION_CHECK(dynamic_cast<ngraph::Node*>(op),
                              (input[i].rank().is_static()),
                              "RNNCellBase supports only static rank for input tensors. Input ",
                              i);
    }

    // Verify input dimension against values provided in spec (LSTMCell_1.md)
    for (size_t i = 0; i < input.size(); i++) {
        if (i == B) {
            // verify only B input dimension which is 1D
            NODE_VALIDATION_CHECK(dynamic_cast<ngraph::Node*>(op),
                                  (input[i].rank().get_length() == 1),
                                  "RNNCellBase B input tensor dimension is not correct.");
        } else {
            // Verify all other input dimensions which are 2D tensor types
            NODE_VALIDATION_CHECK(dynamic_cast<ngraph::Node*>(op),
                                  (input[i].rank().get_length() == 2),
                                  "RNNCellBase input tensor dimension is not correct for ",
                                  i,
                                  " input parameter. Current input length: ",
                                  input[i].rank().get_length(),
                                  ", expected: 2.");
        }
    }

    // Compare input_size dimension for X and W inputs
    const auto& x_pshape = input.at(X);
    const auto& w_pshape = input.at(W);

    NODE_VALIDATION_CHECK(dynamic_cast<ngraph::Node*>(op),
                          (x_pshape[1].compatible(w_pshape[1])),
                          "RNNCellBase mismatched input_size dimension.");
}
}  // namespace util
}  // namespace op
}  // namespace ov
