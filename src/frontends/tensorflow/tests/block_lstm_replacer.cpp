// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/block_lstm_replacer.hpp"

#include <frontend/shared/include/utils.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>

#include "gtest/gtest.h"
#include "helper_ops/block_lstm.hpp"

using namespace std;
using namespace ov;
using namespace opset9;
using namespace element;
using namespace frontend::tensorflow;
using namespace frontend::tensorflow::pass;

namespace {
shared_ptr<Model> gen_model(size_t batch_size, size_t time_len, size_t hidden_size, size_t input_size) {
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

    auto seq_len_max = make_shared<Parameter>(i64, Shape{});
    auto x = make_shared<Parameter>(f32, Shape{time_len, batch_size, input_size});
    auto cs_prev = make_shared<Parameter>(f32, PartialShape::dynamic());
    auto h_prev = make_shared<Parameter>(f32, PartialShape::dynamic());
    auto w = make_shared<Parameter>(f32, PartialShape::dynamic());
    auto wci = make_shared<Parameter>(f32, PartialShape::dynamic());
    auto wcf = make_shared<Parameter>(f32, PartialShape::dynamic());
    auto wco = make_shared<Parameter>(f32, PartialShape::dynamic());
    auto b = make_shared<Parameter>(f32, PartialShape::dynamic());

    auto block_lstm = make_shared<BlockLSTM>(seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, 1.0f, -1.0f, false);
    // block_lstm->output

    // block_lstm->

    // block_lstm->output(6);
    // seq_len_max->output(6);

    // return make_shared<Model>(OutputVector{block_lstm}, )
    /*
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto WRzr = make_shared<Parameter>(f32, Shape{2 * hidden_size, input_size + hidden_size});
    auto Bzr = make_shared<Parameter>(f32, Shape{1, 2 * hidden_size});
    auto WRh = make_shared<Parameter>(f32, Shape{hidden_size, input_size + hidden_size});
    auto Bh = make_shared<Parameter>(f32, Shape{1, hidden_size});
    auto A = make_shared<Parameter>(f32, Shape{batch, 1});
    auto concat_1 = make_shared<Concat>(OutputVector{X, H}, 1);
    auto matmul_1 = make_shared<MatMul>(concat_1, WRzr, false, true);
    auto in_to_activation_1 = make_shared<Add>(matmul_1, Bzr);

    auto sigmoid = make_shared<Sigmoid>(in_to_activation_1);
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto split = make_shared<Split>(sigmoid, axis_1, 2);

    auto multiply_1 = make_shared<Multiply>(split, H);
    auto concat_2 = make_shared<Concat>(OutputVector{X, multiply_1}, 1);
    auto matmul_2 = make_shared<MatMul>(concat_2, WRh, false, true);
    auto in_to_activation_2 = make_shared<Add>(matmul_2, Bh);
    auto tanh = make_shared<Tanh>(in_to_activation_2);

    auto one = make_shared<Constant>(f32, Shape{1}, 1);
    auto subtract_1 = make_shared<Subtract>(one, A);
    auto multiply_2 = make_shared<Multiply>(subtract_1, split->output(1));
    auto subtract_2 = make_shared<Subtract>(one, multiply_2);
    auto multiply_3 = make_shared<Multiply>(subtract_2, tanh);

    auto multiply_4 = make_shared<Multiply>(multiply_2, H);
    auto add = make_shared<Add>(multiply_4, multiply_3);
    return make_shared<Model>(OutputVector{add}, ParameterVector{X, H, WRzr, WRh, Bzr, Bh, A});
    */
    return nullptr;
}

}  // namespace

TEST(BlockLSTMReplacerTest, HiddenStatesOneOutput) {
    ov::pass::Manager pass_manager;
    pass_manager.register_pass<BlockLSTMToLSTMSequenceOneOutput>();
}
