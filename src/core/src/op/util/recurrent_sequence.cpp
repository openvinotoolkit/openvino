// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/recurrent_sequence.hpp"

void ov::op::util::validate_seq_input_rank_dimension(const std::vector<ov::PartialShape>& input) {
    enum { X, initial_hidden_state, sequence_lengths, W, R, B };

    // Verify static ranks for all inputs
    for (size_t i = 0; i < input.size(); i++) {
        OPENVINO_ASSERT((input[i].rank().is_static()), "RNN Sequence supports only static rank for input tensors.");
    }

    for (size_t i = 0; i < input.size(); i++) {
        if (i == B) {
            // verify B input dimension which is 2D
            OPENVINO_ASSERT((input[i].rank().get_length() == 2),
                            "RNN Sequence B input tensor dimension is not correct.");
        } else if (i == sequence_lengths) {
            // verify sequence_length input dimension which is 1D
            OPENVINO_ASSERT((input[i].rank().get_length() == 1),
                            "RNN Sequence sequence_lengths input tensor dimension is not correct.");
        } else {
            // Verify all other input dimensions which are 3D tensor types
            OPENVINO_ASSERT((input[i].rank().get_length() == 3),
                            "RNN Sequence input tensor dimension is not correct for ",
                            i,
                            " input parameter. Current input length: ",
                            input[i].rank().get_length());
        }
    }

    // Compare input_size dimension for X and W inputs
    const auto& x_pshape = input.at(X);
    const auto& w_pshape = input.at(W);

    OPENVINO_ASSERT((x_pshape[2].compatible(w_pshape[2])), "RNN Sequence mismatched input_size dimension.");
}
