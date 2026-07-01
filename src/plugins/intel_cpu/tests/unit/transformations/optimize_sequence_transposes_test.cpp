// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <openvino/core/model.hpp>
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset5_decl.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include <transformations/cpu_opset/common/pass/rnn_sequences_optimization.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"

using namespace testing;
using namespace ov::intel_cpu;

enum class SeqType { LSTM, RNN, GRU };

static size_t getGateCount(SeqType type) {
    switch (type) {
        case SeqType::LSTM: return 4;  // 4*hidden_size
        case SeqType::GRU:  return 3;  // 3*hidden_size
        case SeqType::RNN:  return 1;  // 1*hidden_size
    }
    return 1;
}

static std::shared_ptr<ov::Model> buildOriginalModel(SeqType type, const ov::PartialShape& inputShape) {
    const size_t hidden_size = 128;
    const size_t input_size = 16;
    const size_t gates = getGateCount(type);

    auto X = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputShape);
    auto Y = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden_size});

    auto w_val = std::vector<float>(gates * hidden_size * input_size, 0);
    auto r_val = std::vector<float>(gates * hidden_size * hidden_size, 0);
    auto b_val = std::vector<float>(gates * hidden_size, 0);
    auto W = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, gates * hidden_size, input_size}, w_val);
    auto R = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, gates * hidden_size, hidden_size}, r_val);
    auto B = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, gates * hidden_size}, b_val);

    auto transpose_before_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2});
    auto transpose_before = std::make_shared<ov::opset1::Transpose>(X, transpose_before_const);

    auto seq_lengths = ov::opset1::Constant::create(ov::element::i32, ov::Shape{1}, {2});

    std::shared_ptr<ov::Node> seq_node;
    ov::ParameterVector params;
    if (type == SeqType::LSTM) {
        auto Z = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden_size});
        seq_node = std::make_shared<ov::opset5::LSTMSequence>(transpose_before, Y, Z, seq_lengths, W, R, B, hidden_size,
            ov::op::RecurrentSequenceDirection::FORWARD);
        params = {X, Y, Z};
    } else if (type == SeqType::RNN) {
        seq_node = std::make_shared<ov::opset5::RNNSequence>(transpose_before, Y, seq_lengths, W, R, B, hidden_size,
            ov::op::RecurrentSequenceDirection::FORWARD);
        params = {X, Y};
    } else {
        seq_node = std::make_shared<ov::opset5::GRUSequence>(transpose_before, Y, seq_lengths, W, R, B, hidden_size,
            ov::op::RecurrentSequenceDirection::FORWARD);
        params = {X, Y};
    }

    auto transpose_after_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {2, 1, 0, 3});
    auto transpose_after = std::make_shared<ov::opset1::Transpose>(seq_node->output(0), transpose_after_const);

    auto Y_out = std::make_shared<ov::opset1::Result>(transpose_after);
    auto Ho = std::make_shared<ov::opset1::Result>(seq_node->output(1));
    Y_out->set_friendly_name("Y_out");
    Ho->set_friendly_name("Ho");

    ov::ResultVector results{Y_out, Ho};
    if (type == SeqType::LSTM) {
        auto Co = std::make_shared<ov::opset1::Result>(seq_node->output(2));
        Co->set_friendly_name("Co");
        results.push_back(Co);
    }
    return std::make_shared<ov::Model>(results, params);
}

static std::shared_ptr<ov::Model> buildRefModel(SeqType type, const ov::PartialShape& inputShape) {
    const size_t hidden_size = 128;
    const size_t input_size = 16;
    const size_t gates = getGateCount(type);

    auto X = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputShape);
    auto Y = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden_size});

    auto w_val = std::vector<float>(gates * hidden_size * input_size, 0);
    auto r_val = std::vector<float>(gates * hidden_size * hidden_size, 0);
    auto b_val = std::vector<float>(gates * hidden_size, 0);
    auto W = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, gates * hidden_size, input_size}, w_val);
    auto R = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, gates * hidden_size, hidden_size}, r_val);
    auto B = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, gates * hidden_size}, b_val);

    std::shared_ptr<ov::Node> reshape_before;
    if (inputShape.is_dynamic()) {
        auto data = std::make_shared<ov::opset1::ShapeOf>(X);
        auto reshape_before_pattern = std::make_shared<ov::opset8::Gather>(data,
            ov::opset1::Constant::create(ov::element::i32, {3}, {1, 0, 2}),
            ov::opset1::Constant::create(ov::element::i32, {}, {0}));
        reshape_before = std::make_shared<ov::opset1::Reshape>(X, reshape_before_pattern, false);
    } else {
        auto reshape_before_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{3}, {1, 2, 16});
        reshape_before = std::make_shared<ov::opset1::Reshape>(X, reshape_before_const, false);
    }

    auto seq_lengths = ov::opset1::Constant::create(ov::element::i32, ov::Shape{1}, {2});

    std::shared_ptr<ov::Node> seq_node;
    ov::ParameterVector params;
    if (type == SeqType::LSTM) {
        auto Z = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden_size});
        seq_node = std::make_shared<ov::opset5::LSTMSequence>(reshape_before, Y, Z, seq_lengths, W, R, B, hidden_size,
            ov::op::RecurrentSequenceDirection::FORWARD);
        params = {X, Y, Z};
    } else if (type == SeqType::RNN) {
        seq_node = std::make_shared<ov::opset5::RNNSequence>(reshape_before, Y, seq_lengths, W, R, B, hidden_size,
            ov::op::RecurrentSequenceDirection::FORWARD);
        params = {X, Y};
    } else {
        seq_node = std::make_shared<ov::opset5::GRUSequence>(reshape_before, Y, seq_lengths, W, R, B, hidden_size,
            ov::op::RecurrentSequenceDirection::FORWARD);
        params = {X, Y};
    }

    auto reshape_after_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {2, 1, 1, 128});
    auto reshape_after = std::make_shared<ov::opset1::Reshape>(seq_node->output(0), reshape_after_const, false);

    auto Y_out = std::make_shared<ov::opset1::Result>(reshape_after);
    auto Ho = std::make_shared<ov::opset1::Result>(seq_node->output(1));
    Y_out->set_friendly_name("Y_out");
    Ho->set_friendly_name("Ho");

    ov::ResultVector results{Y_out, Ho};
    if (type == SeqType::LSTM) {
        auto Co = std::make_shared<ov::opset1::Result>(seq_node->output(2));
        Co->set_friendly_name("Co");
        results.push_back(Co);
    }
    return std::make_shared<ov::Model>(results, params);
}

using OptimizeSeqTransposeParams = std::tuple<SeqType, ov::PartialShape>;

class OptimizeSequenceTransposesTests : public TransformationTestsF,
                                        public WithParamInterface<OptimizeSeqTransposeParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OptimizeSeqTransposeParams>& info) {
        const auto& [seqType, shape] = info.param;
        std::ostringstream ss;
        switch (seqType) {
            case SeqType::LSTM: ss << "LSTM"; break;
            case SeqType::RNN:  ss << "RNN"; break;
            case SeqType::GRU:  ss << "GRU"; break;
        }
        ss << (shape.is_dynamic() ? "_Dynamic" : "_Static");
        return ss.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        disable_rt_info_check();
        const auto& [seqType, shape] = GetParam();
        model = buildOriginalModel(seqType, shape);
        model_ref = buildRefModel(seqType, shape);
        switch (seqType) {
            case SeqType::LSTM: manager.register_pass<OptimizeLSTMSequenceTransposes>(); break;
            case SeqType::RNN:  manager.register_pass<OptimizeRNNSequenceTransposes>(); break;
            case SeqType::GRU:  manager.register_pass<OptimizeGRUSequenceTransposes>(); break;
        }
    }
};

TEST_P(OptimizeSequenceTransposesTests, CompareWithRef) {}

INSTANTIATE_TEST_SUITE_P(TransformationTests, OptimizeSequenceTransposesTests,
    ::testing::Combine(
        ::testing::Values(SeqType::LSTM, SeqType::RNN, SeqType::GRU),
        ::testing::Values(ov::PartialShape{2, 1, 16}, ov::PartialShape{2, -1, -1})),
    OptimizeSequenceTransposesTests::getTestCaseName);
