// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class GRUBlockCell : public InternalOperation {
public:
    OPENVINO_OP("GRUBlockCell", "ov::frontend::tensorflow::util", InternalOperation);

    GRUBlockCell(const Output<Node>& x,
                 const Output<Node>& h_prev,
                 const Output<Node>& w_ru,
                 const Output<Node>& w_c,
                 const Output<Node>& b_ru,
                 const Output<Node>& b_c,
                 const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{x, h_prev, w_ru, w_c, b_ru, b_c}, 4, "GRUBlockCell"),
          m_hidden_size(ov::Dimension::dynamic()) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // GRUBlockCell computes the GRU cell forward propagation for 1 time step
        // Inputs:
        // 0) x: Input to the GRU cell
        // 1) h_prev: State input from the previous GRU cell
        // 2) w_ru: Weight matrix for the reset and update gate
        // 3) w_c: Weight matrix for the cell connection gate
        // 4) b_ru: Bias vector for the reset and update gate
        // 5) b_c: Bias vector for the cell connection gate
        //
        // Outputs:
        // 0) r: Output of the reset gate
        // 1) u: Output of the update gate
        // 2) c: Output of the cell connection gate
        // 3) h: Current state of the GRU cell

        // try to deduce static hidden_size
        // 1. use h_prev shape
        auto h_prev_shape = get_input_partial_shape(1);
        auto h_prev_rank = h_prev_shape.rank();
        if (h_prev_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(h_prev_rank.get_length() == 2,
                                          "Internal error in OpenVINO TensorFlow Frontend: initial hidden state for "
                                          "GRUBlockCell must be of rank equal to 2.");
            m_hidden_size = h_prev_shape[1].is_static() ? h_prev_shape[1].get_length() : m_hidden_size;
        }
        // 2. use w_ru shape
        auto w_ru_shape = get_input_partial_shape(2);
        auto w_ru_rank = w_ru_shape.rank();
        if (w_ru_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                w_ru_rank.get_length() == 2,
                "Internal error in OpenVINO TensorFlow Frontend: weights for GRUBlockCell must be of rank equal to 2.");
            m_hidden_size = w_ru_shape[1].is_static() ? w_ru_shape[1].get_length() / 2 : m_hidden_size;
        }
        // 3. use w_c shape
        auto w_c_shape = get_input_partial_shape(3);
        auto w_c_rank = w_c_shape.rank();
        if (w_c_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                w_c_rank.get_length() == 2,
                "Internal error in OpenVINO TensorFlow Frontend: weights for GRUBlockCell must be of rank equal to 2.");
            m_hidden_size = w_c_shape[1].is_static() ? w_c_shape[1].get_length() : m_hidden_size;
        }
        // 3. use b_ru shape
        auto b_ru_shape = get_input_partial_shape(4);
        auto b_ru_rank = b_ru_shape.rank();
        if (b_ru_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                b_ru_rank.get_length() == 1,
                "Internal error in OpenVINO TensorFlow Frontend: bias for GRUBlockCell must be of rank equal to 1.");
            m_hidden_size = b_ru_shape[0].is_static() ? b_ru_shape[0].get_length() / 2 : m_hidden_size;
        }
        // 4. use b_c shape
        auto b_c_shape = get_input_partial_shape(5);
        auto b_c_rank = b_c_shape.rank();
        if (b_c_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                b_c_rank.get_length() == 1,
                "Internal error in OpenVINO TensorFlow Frontend: bias for GRUBlockCell must be of rank equal to 1.");
            m_hidden_size = b_c_shape[0].is_static() ? b_c_shape[0].get_length() : m_hidden_size;
        }

        // set the defined shape only for the fourth output since
        // OpenVINO GRUCell supports hidden state output
        auto x_type = get_input_element_type(0);
        set_output_type(0, x_type, ov::PartialShape::dynamic());
        set_output_type(1, x_type, ov::PartialShape::dynamic());
        set_output_type(2, x_type, ov::PartialShape::dynamic());
        set_output_type(3, x_type, h_prev_shape);
    }

    ov::Dimension get_hidden_size() const {
        // TODO: it must be deleted once hidden_size is gone from attributes
        return m_hidden_size;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 6,
                                      "[TensorFlow Frontend] internal error: GRUBlockCell expects 6 inputs");
        auto gru_block_cell_node =
            std::make_shared<GRUBlockCell>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], m_decoder);
        gru_block_cell_node->set_attrs(get_attrs());
        return gru_block_cell_node;
    }

private:
    ov::Dimension m_hidden_size;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
