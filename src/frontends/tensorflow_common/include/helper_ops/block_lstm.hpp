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

class BlockLSTM : public InternalOperation {
public:
    OPENVINO_OP("BlockLSTM", "ov::frontend::tensorflow::util", InternalOperation);

    BlockLSTM(const Output<Node>& seq_len_max,
              const Output<Node>& x,
              const Output<Node>& cs_prev,
              const Output<Node>& h_prev,
              const Output<Node>& w,
              const Output<Node>& wci,
              const Output<Node>& wcf,
              const Output<Node>& wco,
              const Output<Node>& b,
              float forget_bias,
              float cell_clip,
              bool use_peephole,
              const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder,
                            OutputVector{seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b},
                            7,
                            "BlockLSTM"),
          m_hidden_size(ov::Dimension::dynamic()),
          m_forget_bias(forget_bias),
          m_cell_clip(cell_clip),
          m_use_peephole(use_peephole) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // BlockLSTM description
        //
        // Inputs:
        // 0) seq_len_max	A Tensor of type int64. Maximum time length actually used by this input. Outputs are
        // padded with zeros beyond this length.
        // 1) x A Tensor. Must be one of the following types: half, float32. The sequence input to the LSTM, shape
        // (timelen, batch_size, num_inputs).
        // 2) cs_prev A Tensor. Must have the same type as x. Value of the initial cell state.
        // 3) h_prev A Tensor. Must have the same type as x. Initial output of cell (to be used for peephole).
        // 4) w	A Tensor. Must have the same type as x. The weight matrix.
        // 5) wci A Tensor. Must have the same type as x. The weight matrix for input gate peephole connection.
        // 6) wcf A Tensor. Must have the same type as x. The weight matrix for forget gate peephole connection.
        // 7) wco A Tensor. Must have the same type as x. The weight matrix for output gate peephole connection.
        // 8) b	A Tensor. Must have the same type as x. The bias vector.
        //
        // Attributes:
        // forget_bias	An optional float. Defaults to 1. The forget gate bias.
        // cell_clip	An optional float. Defaults to 3. Value to clip the 'cs' value to.
        // use_peephole	An optional bool. Defaults to False. Whether to use peephole weights.
        //
        // Outputs:
        // 0) i A Tensor. Has the same type as x.
        // 1) cs A Tensor. Has the same type as x.
        // 2) f A Tensor. Has the same type as x.
        // 3) o A Tensor. Has the same type as x
        // 4) ci A Tensor. Has the same type as x.
        // 5) co A Tensor. Has the same type as x.
        // 6) h A Tensor. Has the same type as x.

        // extract time_len, batch_size
        auto time_len = ov::Dimension::dynamic();
        auto batch_size = ov::Dimension::dynamic();
        auto x_shape = get_input_partial_shape(1);
        auto x_type = get_input_element_type(1);
        auto x_rank = x_shape.rank();
        if (x_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                x_rank.get_length() == 3,
                "Internal error in OpenVINO TensorFlow Frontend: input data for BlockLSTM must be of rank equal to 3.");
            time_len = x_shape[0].is_static() ? x_shape[0].get_length() : time_len;
            batch_size = x_shape[1].is_static() ? x_shape[1].get_length() : batch_size;
        }

        // extract hidden_size
        auto w_shape = get_input_partial_shape(4);
        auto w_rank = w_shape.rank();
        auto b_shape = get_input_partial_shape(8);
        auto b_rank = b_shape.rank();
        if (w_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                w_rank.get_length() == 2,
                "Internal error in OpenVINO TensorFlow Frontend: weights for BlockLSTM must be of rank equal to 2.");
            m_hidden_size = w_shape[1].is_static() ? w_shape[1].get_length() / 4 : ov::Dimension::dynamic();
        }
        if (b_rank.is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(
                b_rank.get_length() == 1,
                "Internal error in OpenVINO TensorFlow Frontend: weights for BlockLSTM must be of rank equal to 2.");
            m_hidden_size = b_shape[0].is_static() ? b_shape[0].get_length() / 4 : m_hidden_size;
        }

        // set shapes of dynamic rank for inputs except 1 and 6 which OpenVINO LSTMSequence supports now
        // other outputs are out of scope
        set_output_type(0, x_type, ov::PartialShape::dynamic());
        set_output_type(1, x_type, ov::PartialShape{time_len, batch_size, m_hidden_size});
        set_output_type(2, x_type, ov::PartialShape::dynamic());
        set_output_type(3, x_type, ov::PartialShape::dynamic());
        set_output_type(4, x_type, ov::PartialShape::dynamic());
        set_output_type(5, x_type, ov::PartialShape::dynamic());
        set_output_type(6, x_type, ov::PartialShape{time_len, batch_size, m_hidden_size});
    }

    float get_forget_bias() const {
        return m_forget_bias;
    }

    float get_cell_clip() const {
        return m_cell_clip;
    }

    bool get_use_peephole() const {
        return m_use_peephole;
    }

    ov::Dimension get_hidden_size() const {
        // TODO: it must be deleted once hidden_size is gone from attributes
        return m_hidden_size;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 9,
                                      "[TensorFlow Frontend] internal error: BlockLSTM expects 9 inputs");
        auto block_lstm_node = std::make_shared<BlockLSTM>(inputs[0],
                                                           inputs[1],
                                                           inputs[2],
                                                           inputs[3],
                                                           inputs[4],
                                                           inputs[5],
                                                           inputs[6],
                                                           inputs[7],
                                                           inputs[8],
                                                           m_forget_bias,
                                                           m_cell_clip,
                                                           m_use_peephole,
                                                           m_decoder);
        block_lstm_node->set_attrs(get_attrs());
        return block_lstm_node;
    }

private:
    ov::Dimension m_hidden_size;
    float m_forget_bias;
    float m_cell_clip;
    bool m_use_peephole;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
