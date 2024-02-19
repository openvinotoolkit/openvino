// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/block_lstm_replacer.hpp"

#include <gtest/gtest.h>

#include "conversion_with_reference.hpp"
#include "helper_ops/block_lstm.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace element;
using namespace frontend::tensorflow;
using namespace frontend::tensorflow::pass;

namespace {
shared_ptr<Model> gen_model(Dimension batch_size,
                            Dimension time_len,
                            int64_t hidden_size,
                            Dimension input_size,
                            float forget_bias,
                            float cell_clip,
                            bool use_peephole,
                            bool with_two_outputs = false) {
    auto seq_len_max = make_shared<v0::Parameter>(i64, Shape{});
    auto x = make_shared<v0::Parameter>(f32, PartialShape{time_len, batch_size, input_size});
    auto cs_prev = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto h_prev = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto w = make_shared<v0::Parameter>(f32, PartialShape{Dimension::dynamic(), 4 * hidden_size});
    auto wci = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto wcf = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto wco = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b = make_shared<v0::Parameter>(f32, PartialShape::dynamic());

    auto block_lstm = make_shared<
        BlockLSTM>(seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, forget_bias, cell_clip, use_peephole);

    if (with_two_outputs) {
        auto prev_cell_states = make_shared<v0::Constant>(
            ov::element::f32,
            ov::Shape{1, static_cast<std::size_t>(batch_size.get_length()), static_cast<std::size_t>(hidden_size)},
            0);
        auto concat = make_shared<v0::Concat>(OutputVector{prev_cell_states, block_lstm->output(1)}, 0);
        auto indices_const = make_shared<v0::Constant>(ov::element::i32,
                                                       ov::Shape{2},
                                                       vector<int32_t>{static_cast<int32_t>(time_len.get_length()), 0});
        auto gather_nd = make_shared<v8::GatherND>(concat, indices_const);
        return make_shared<Model>(OutputVector{gather_nd->output(0), block_lstm->output(6)},
                                  ParameterVector{seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b});
    }

    return make_shared<Model>(OutputVector{block_lstm->output(6)},
                              ParameterVector{seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b});
}

shared_ptr<Model> gen_model_ref(Dimension m_batch_size,
                                Dimension m_time_len,
                                int64_t m_hidden_size,
                                Dimension m_input_size,
                                float forget_bias,
                                bool with_two_outputs = false) {
    auto seq_len_max = make_shared<v0::Parameter>(i64, Shape{});
    auto x = make_shared<v0::Parameter>(f32, PartialShape{m_time_len, m_batch_size, m_input_size});
    auto cs_prev = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto h_prev = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto weights = make_shared<v0::Parameter>(f32, PartialShape{Dimension::dynamic(), 4 * m_hidden_size});
    auto bias = make_shared<v0::Parameter>(f32, PartialShape::dynamic());

    auto x_shape = make_shared<v3::ShapeOf>(x, element::i64);
    auto ss_start = make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto ss_stop = make_shared<v0::Constant>(element::i64, Shape{1}, 3);
    auto ss_step = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto input_size = make_shared<v1::StridedSlice>(x_shape,
                                                    ss_start,
                                                    ss_stop,
                                                    ss_step,
                                                    std::vector<int64_t>{0},
                                                    std::vector<int64_t>{0});

    // retrieve the batch size
    // now x is in a format [time_len, batch_size, input_size]
    auto ss_start2 = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto ss_stop2 = make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto batch_size = make_shared<v1::StridedSlice>(x_shape,
                                                    ss_start2,
                                                    ss_stop2,
                                                    ss_step,
                                                    std::vector<int64_t>{0},
                                                    std::vector<int64_t>{0});
    auto hidden_size_const = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{m_hidden_size});

    // adjust weights and bias
    // 1. reshape weights and bias to highlight channel dimension
    auto new_weight_shape = make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 4, -1});
    auto weight_reshape = make_shared<v1::Reshape>(weights, new_weight_shape, true);
    auto new_bias_shape = make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{4, -1});
    auto bias_reshape = make_shared<v1::Reshape>(bias, new_bias_shape, true);
    // 2. reorder gates icfo --> fico for both weights and biases
    auto reorder_const = make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{2, 0, 1, 3});
    auto weights_axis = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto weights_reorder = make_shared<v8::Gather>(weight_reshape, reorder_const, weights_axis);
    auto bias_axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto bias_reorder = make_shared<v8::Gather>(bias_reshape, reorder_const, bias_axis);
    // 3. shift_const.value should be added to the first 1 / 4th part of the biases(f - gate : 0)
    auto shift_const = make_shared<v0::Constant>(element::f32, Shape{}, forget_bias);
    auto bias_split_lens = make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 3});
    auto bias_split = make_shared<v1::VariadicSplit>(bias_reorder, bias_axis, bias_split_lens);
    auto bias_first_shift = make_shared<v1::Add>(bias_split->output(0), shift_const);
    auto bias_shift = make_shared<v0::Concat>(OutputVector{bias_first_shift, bias_split->output(1)}, 0);
    // 4. return to the original shapes
    auto new_weight_shape2 = make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
    auto weight_reshape2 = make_shared<v1::Reshape>(weights_reorder, new_weight_shape2, true);
    // 5. normalize weights and bias
    auto transpose_order = make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
    auto new_bias_shape2 = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto weights_normalize = make_shared<v1::Transpose>(weight_reshape2, transpose_order);
    auto bias_normalized = make_shared<v1::Reshape>(bias_shift, new_bias_shape2, true);
    // 6. split weights into W and R inputs
    auto WR_split_axis = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto WR_split_lens = make_shared<v0::Concat>(OutputVector{input_size, hidden_size_const}, 0);
    auto WR_split = make_shared<v1::VariadicSplit>(weights_normalize, WR_split_axis, WR_split_lens);
    // 7. unsqueeze weights and bias to have a dimension for a number of directions
    auto num_direct_axis = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto W = make_shared<v0::Unsqueeze>(WR_split->output(0), num_direct_axis);
    auto R = make_shared<v0::Unsqueeze>(WR_split->output(1), num_direct_axis);
    auto B = make_shared<v0::Unsqueeze>(bias_normalized, num_direct_axis);

    // normalize initial hidden and cell states
    auto unsqueeze_axis = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto init_hidden_state = make_shared<v0::Unsqueeze>(h_prev, unsqueeze_axis);
    auto init_cell_state = make_shared<v0::Unsqueeze>(cs_prev, unsqueeze_axis);

    // prepare sequence length input for LSTMSequence
    auto seq_len_max_adjusted = make_shared<v3::Broadcast>(seq_len_max, batch_size);

    // prepare input data since LSTMSequence accept it in a format [batch_size, time_len, input_size]
    auto x_order = make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto x_adjusted = make_shared<v1::Transpose>(x, x_order);

    // create LSTMSequence node and reconnect inputs and normalized weights and bias
    auto lstm_sequence = make_shared<v5::LSTMSequence>(x_adjusted,
                                                       init_hidden_state,
                                                       init_cell_state,
                                                       seq_len_max_adjusted,
                                                       W,
                                                       R,
                                                       B,
                                                       m_hidden_size,
                                                       v5::LSTMSequence::direction::FORWARD);

    // adjust output of concatenated of hidden states from LSTMSequence to have it in a format [time_len,
    // batch_size, hidden_size]
    // 1. squeeze extra dimension - num_directions
    auto squeeze_axis = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto squeeze_output_hidden_states = make_shared<v0::Squeeze>(lstm_sequence->output(0), squeeze_axis);
    // 2. transpose the output to rotate batch and time dimensions
    auto output_hidden_states_order = make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto output_hidden_states = make_shared<v1::Transpose>(squeeze_output_hidden_states, output_hidden_states_order);

    if (with_two_outputs) {
        // adjust output with the last state cell and connect to the main graph
        // squeeze extra dimension - num_directions
        auto squeeze_axis = make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
        auto squeeze_last_state_cell = make_shared<v0::Squeeze>(lstm_sequence->output(2), squeeze_axis);
        return make_shared<Model>(OutputVector{squeeze_last_state_cell->output(0), output_hidden_states->output(0)},
                                  ParameterVector{seq_len_max, x, cs_prev, h_prev, weights, bias});
    }

    return make_shared<Model>(OutputVector{output_hidden_states->output(0)},
                              ParameterVector{seq_len_max, x, cs_prev, h_prev, weights, bias});
}

}  // namespace

TEST_F(FrontEndConversionWithReferenceTestsF, BlockLSTMReplacerWithHiddenOutput) {
    {
        model = gen_model(2, 10, 120, 20, 1.0f, -1.0f, false);
        manager.register_pass<BlockLSTMReplacer>();
    }
    { model_ref = gen_model_ref(2, 10, 120, 20, 1.0f); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, BlockLSTMReplacerWithHiddenOutputAndLastCellState) {
    {
        model = gen_model(2, 10, 120, 20, 1.0f, -1.0f, false, true);
        manager.register_pass<BlockLSTMReplacer>();
    }
    { model_ref = gen_model_ref(2, 10, 120, 20, 1.0f, true); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, BlockLSTMReplacerWithPeepHole) {
    {
        model = gen_model(2, 10, 120, 20, 1.0f, -1.0f, true);
        manager.register_pass<BlockLSTMReplacer>();
    }
    {
        // the transformation is not applied for the peep hole case
        model_ref = gen_model(2, 10, 120, 20, 1.0f, -1.0f, true);
    }
}
