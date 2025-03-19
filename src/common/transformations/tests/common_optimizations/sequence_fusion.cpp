// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sequence_fusion.hpp"

#include <gtest/gtest.h>

#include <queue>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset9.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

using namespace ov;
using namespace std;
using namespace testing;
using namespace ov::opset9;
using namespace ov::element;

namespace {
enum class RNN_TYPE { LSTM_v0, LSTM_v4, GRU, RNN, AUGRU };

int get_gate_by_rnn_type(RNN_TYPE rnn_type) {
    int gate = 1;
    if (rnn_type == RNN_TYPE::LSTM_v4 || rnn_type == RNN_TYPE::LSTM_v0) {
        gate = 4;
    } else if (rnn_type == RNN_TYPE::GRU || rnn_type == RNN_TYPE::AUGRU) {
        gate = 3;
    } else if (rnn_type == RNN_TYPE::RNN) {
        gate = 1;
    }
    return gate;
}

OutputVector create_cell(RNN_TYPE rnn_type,
                         const shared_ptr<Node>& X,
                         const shared_ptr<Node>& H,
                         const shared_ptr<Node>& C,
                         const shared_ptr<Node>& W,
                         const shared_ptr<Node>& R,
                         const shared_ptr<Node>& B,
                         const shared_ptr<Node>& A,
                         size_t hidden_size,
                         int64_t cells_cnt) {
    shared_ptr<Node> cell;
    Output<Node> cur_H = H;
    Output<Node> cur_C = C;
    OutputVector hidden_vec;
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);

    for (int i = 0; i < cells_cnt; ++i) {
        if (rnn_type == RNN_TYPE::LSTM_v4) {
            cell = make_shared<LSTMCell>(X, cur_H, cur_C, W, R, B, hidden_size);
            cur_C = cell->output(1);
        } else if (rnn_type == RNN_TYPE::LSTM_v0) {
            cell =
                make_shared<opset3::LSTMCell>(X, cur_H, cur_C, W, R, B, hidden_size, ov::op::LSTMWeightsFormat::FICO);
            cur_C = cell->output(1);
        } else if (rnn_type == RNN_TYPE::GRU) {
            cell = make_shared<GRUCell>(X, cur_H, W, R, B, hidden_size);
        } else if (rnn_type == RNN_TYPE::RNN) {
            cell = make_shared<RNNCell>(X, cur_H, W, R, B, hidden_size);
        } else if (rnn_type == RNN_TYPE::AUGRU) {
            cell = make_shared<op::internal::AUGRUCell>(X, cur_H, W, R, B, A, hidden_size);
        }
        cur_H = cell->output(0);
        hidden_vec.push_back(make_shared<Unsqueeze>(cur_H, axis_1));
    }
    auto concat = make_shared<Concat>(hidden_vec, 1);
    OutputVector outputs = {concat->output(0)};
    auto cell_outputs = cell->outputs();
    outputs.insert(outputs.end(), cell_outputs.begin(), cell_outputs.end());
    return outputs;
}

shared_ptr<Model> gen_model(RNN_TYPE rnn_type, size_t batch, size_t hidden_size, size_t input_size, int64_t cells_cnt) {
    int gate = get_gate_by_rnn_type(rnn_type);
    auto X = make_shared<Parameter>(f32, Shape{batch, input_size});
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto C = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto W = make_shared<Parameter>(f32, Shape{gate * hidden_size, input_size});
    auto R = make_shared<Parameter>(f32, Shape{gate * hidden_size, hidden_size});
    auto B = make_shared<Parameter>(f32, Shape{gate * hidden_size});
    auto A = make_shared<Parameter>(f32, Shape{batch, 1});

    auto outputs = create_cell(rnn_type, X, H, C, W, R, B, A, hidden_size, cells_cnt);
    ParameterVector params = {X, H, W, R, B};
    if (rnn_type == RNN_TYPE::LSTM_v4 || rnn_type == RNN_TYPE::LSTM_v0) {
        params.push_back(C);
    } else if (rnn_type == RNN_TYPE::AUGRU) {
        params.push_back(A);
    }
    return make_shared<Model>(outputs, params);
}

shared_ptr<Model> gen_reference(RNN_TYPE rnn_type,
                                size_t batch,
                                size_t hidden_size,
                                size_t input_size,
                                int64_t cells_cnt) {
    int gate = get_gate_by_rnn_type(rnn_type);
    auto axis_0 = make_shared<Constant>(i64, Shape{}, 0);
    auto axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto seq_len = make_shared<Constant>(i64, Shape{batch}, cells_cnt);

    auto X = make_shared<Parameter>(f32, Shape{batch, input_size});
    auto H = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto C = make_shared<Parameter>(f32, Shape{batch, hidden_size});
    auto W = make_shared<Parameter>(f32, Shape{gate * hidden_size, input_size});
    auto R = make_shared<Parameter>(f32, Shape{gate * hidden_size, hidden_size});
    auto B = make_shared<Parameter>(f32, Shape{gate * hidden_size});
    auto A = make_shared<Parameter>(f32, Shape{batch, 1});

    ParameterVector params = {X, H, W, R, B};
    if (rnn_type == RNN_TYPE::LSTM_v4 || rnn_type == RNN_TYPE::LSTM_v0) {
        params.push_back(C);
    } else if (rnn_type == RNN_TYPE::AUGRU) {
        params.push_back(A);
    }
    auto unH = make_shared<Unsqueeze>(H, axis_1);
    auto unC = make_shared<Unsqueeze>(C, axis_1);
    auto unW = make_shared<Unsqueeze>(W, axis_0);
    auto unR = make_shared<Unsqueeze>(R, axis_0);
    auto unB = make_shared<Unsqueeze>(B, axis_0);

    OutputVector in_X;
    OutputVector in_A;
    for (int i = 0; i < cells_cnt; ++i) {
        in_X.push_back(make_shared<Unsqueeze>(X, axis_1));
        in_A.push_back(make_shared<Unsqueeze>(A, axis_1));
    }
    auto concat_X = make_shared<Concat>(in_X, 1);
    auto concat_A = make_shared<Concat>(in_A, 1);

    shared_ptr<Node> seq;
    if (rnn_type == RNN_TYPE::LSTM_v4 || rnn_type == RNN_TYPE::LSTM_v0) {
        seq = make_shared<LSTMSequence>(concat_X,
                                        unH,
                                        unC,
                                        seq_len,
                                        unW,
                                        unR,
                                        unB,
                                        hidden_size,
                                        op::RecurrentSequenceDirection::FORWARD);
    } else if (rnn_type == RNN_TYPE::GRU) {
        seq = make_shared<GRUSequence>(concat_X,
                                       unH,
                                       seq_len,
                                       unW,
                                       unR,
                                       unB,
                                       hidden_size,
                                       op::RecurrentSequenceDirection::FORWARD);
    } else if (rnn_type == RNN_TYPE::RNN) {
        seq = make_shared<RNNSequence>(concat_X,
                                       unH,
                                       seq_len,
                                       unW,
                                       unR,
                                       unB,
                                       hidden_size,
                                       op::RecurrentSequenceDirection::FORWARD);
    } else if (rnn_type == RNN_TYPE::AUGRU) {
        seq = make_shared<op::internal::AUGRUSequence>(concat_X, unH, seq_len, unW, unR, unB, concat_A, hidden_size);
    }

    auto squeeze_H = make_shared<Squeeze>(seq->output(0), axis_1);
    auto _axis_1 = make_shared<Constant>(i64, Shape{}, 1);
    auto split = make_shared<Split>(squeeze_H, axis_1, cells_cnt);

    OutputVector in_vec;
    for (size_t i = 0; i < split->outputs().size(); ++i) {
        auto squeeze = make_shared<Squeeze>(split->output(i), axis_1);
        in_vec.push_back(make_shared<Unsqueeze>(squeeze, _axis_1));
    }
    auto concat = make_shared<Concat>(in_vec, 1);

    auto squeeze_Ht = make_shared<Squeeze>(seq->output(1), axis_1);
    OutputVector outputs = {concat->output(0), squeeze_Ht};
    if (rnn_type == RNN_TYPE::LSTM_v4 || rnn_type == RNN_TYPE::LSTM_v0) {
        auto squeeze_Ct = make_shared<Squeeze>(seq->output(2), axis_1);
        outputs.push_back(squeeze_Ct);
    }
    return make_shared<Model>(outputs, params);
}
}  // namespace

struct SequenceFusionParams {
    RNN_TYPE rnn_type;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    int64_t cell_cnt;
};

class SequenceFusionTest : public WithParamInterface<SequenceFusionParams>, public TransformationTestsF {};

TEST_P(SequenceFusionTest, SequencePattern) {
    const auto& p = GetParam();
    {
        model = gen_model(p.rnn_type, p.batch, p.hidden_size, p.input_size, p.cell_cnt);
        manager.register_pass<pass::SequenceFusion>();
    }

    // the transformation won't be applied for single cell
    if (p.cell_cnt > 1) {
        model_ref = gen_reference(p.rnn_type, p.batch, p.hidden_size, p.input_size, p.cell_cnt);
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

static const std::vector<SequenceFusionParams> params = {
    SequenceFusionParams{RNN_TYPE::LSTM_v4, 2, 128, 32, 1},
    SequenceFusionParams{RNN_TYPE::LSTM_v4, 2, 128, 32, 10},
    SequenceFusionParams{RNN_TYPE::LSTM_v0, 2, 128, 32, 1},
    SequenceFusionParams{RNN_TYPE::LSTM_v0, 2, 128, 32, 10},
    SequenceFusionParams{RNN_TYPE::GRU, 2, 128, 32, 1},
    SequenceFusionParams{RNN_TYPE::GRU, 2, 128, 32, 10},
    SequenceFusionParams{RNN_TYPE::RNN, 2, 128, 32, 1},
    SequenceFusionParams{RNN_TYPE::RNN, 2, 128, 32, 10},
    SequenceFusionParams{RNN_TYPE::AUGRU, 2, 128, 32, 1},
    SequenceFusionParams{RNN_TYPE::AUGRU, 2, 128, 32, 10},
};

INSTANTIATE_TEST_SUITE_P(SequenceFusionTest, SequenceFusionTest, ValuesIn(params));
