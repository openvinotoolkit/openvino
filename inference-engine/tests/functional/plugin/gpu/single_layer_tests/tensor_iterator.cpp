// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include <gpu/gpu_config.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include "common_test_utils/test_constants.hpp"
#include "ie_api.h"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using Config = std::pair<std::string, std::map<std::string, std::string>>;

namespace LayerTestsDefinitions {

using TensorIteratorWithConfigParams = typename std::tuple<
        size_t,                                 // seq_lengths
        size_t,                                 // batch
        size_t,                                 // hidden size
        // todo: fix. input size hardcoded to 10 due to limitation (10 args) of gtests Combine() func.
        //size_t,                               // input size
        size_t,                                 // sequence axis
        float,                                  // clip
        ngraph::helpers::TensorIteratorBody,    // body type
        ngraph::op::RecurrentSequenceDirection, // direction
        InferenceEngine::Precision,             // Network precision
        Config>;                                // Target device name & Configuration

class TensorIteratorWithConfigTest : public testing::WithParamInterface<TensorIteratorWithConfigParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorWithConfigParams> &obj) {
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        size_t sequence_axis;
        ngraph::helpers::TensorIteratorBody ti_body;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::pair<std::string, std::map<std::string, std::string>> config;
        std::tie(seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, netPrecision,
                 config) = obj.param;
        std::vector<std::vector<size_t>> inputShapes = {};

        switch (ti_body) {
            case ngraph::helpers::TensorIteratorBody::LSTM:
                inputShapes = {
                        {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                                {4 * hidden_size, hidden_size}, {4 * hidden_size}},
                };
                break;
            case ngraph::helpers::TensorIteratorBody::GRU:
                inputShapes = {
                        {{batch, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                break;
            case ngraph::helpers::TensorIteratorBody::RNN:
                inputShapes = {{batch, input_size}, {batch, hidden_size},
                               {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
                break;
        }

        std::ostringstream result;
        result << "seq_len=" << seq_lengths << "_";
        result << "seq_len_axis=" << sequence_axis << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "TensorIteratorBody=" << ti_body << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        if (!config.first.empty()) {
            result << "config=";
            for (const auto& c : config.second) {
                result << '(' << c.first << ',';
                result << c.second << ')';
            }
            result << '_';
        }
        result << "targetDevice=" << config.first << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        size_t sequence_axis;
        ngraph::helpers::TensorIteratorBody ti_body;
        float clip;
        ngraph::op::RecurrentSequenceDirection direction;
        std::pair<std::string, std::map<std::string, std::string>> config;
        InferenceEngine::Precision netPrecision;
        std::tie(seq_lengths, batch, hidden_size, sequence_axis, clip, ti_body, direction, netPrecision,
                 config) = this->GetParam();
        targetDevice = config.first;
        configuration = config.second;
        std::vector<std::vector<size_t>> inputShapes;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto tensor_iterator = std::make_shared<ngraph::opset5::TensorIterator>();

        // Each case consist of 3 steps:
        // 1. Create TensorIterator body.
        // 2. Set PortMap
        // 3. Create outer function
        auto axis = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(sequence_axis)});
        switch (ti_body) {
            case ngraph::helpers::TensorIteratorBody::LSTM: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                                {4 * hidden_size, hidden_size}, {4 * hidden_size}},
                };
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2]});
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};
                ngraph::OutputVector out_vector = {squeeze, body_params[1], body_params[2]};
                auto lstm_cell = ngraph::builder::makeLSTM(out_vector, WRB, hidden_size, {"sigmoid", "tanh", "tanh"}, {}, {}, clip);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(lstm_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(unsqueeze),
                                             std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                             std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "lstm_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[0], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[0], -1, -1, 1, 0, sequence_axis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[1]);
                tensor_iterator->set_merged_input(body_params[2], outer_params[2], results[2]);
                tensor_iterator->get_iter_value(results[1]);
                tensor_iterator->get_iter_value(results[2]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1),
                                                                                   tensor_iterator->output(2)}, outer_params);
                break;
            }
            case ngraph::helpers::TensorIteratorBody::GRU: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
                std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1]};
                auto gru_cell = ngraph::builder::makeGRU(out_vector, WRB, hidden_size, {"sigmoid", "tanh"},
                                                         {}, {}, clip, false);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(gru_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0)),
                                             std::make_shared<ngraph::opset1::Result>(unsqueeze)};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "gru_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, sequence_axis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            }
            case ngraph::helpers::TensorIteratorBody::RNN: {
                inputShapes = {{batch, seq_lengths, input_size},
                               {batch,       hidden_size},
                               {hidden_size, input_size},
                               {hidden_size, hidden_size},
                               {hidden_size}};
                if (sequence_axis == 0) {
                    // swap batch and seq_lengths
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});

                // 1. Create TensorIterator body.
                inputShapes[0][sequence_axis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
                std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1]};
                auto rnn_cell = ngraph::builder::makeRNN(out_vector, WRB, hidden_size, {"tanh"}, {}, {}, clip);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(rnn_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(rnn_cell),
                                             std::make_shared<ngraph::opset1::Result>(unsqueeze)};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "rnn_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, sequence_axis);
                } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, sequence_axis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, sequence_axis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            }
        }
    }
};

TEST_P(TensorIteratorWithConfigTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    INSTANTIATE_TEST_CASE_P(smoke_TensorIteratorCommon, TensorIteratorWithConfigTest,
        ::testing::Combine(
            ::testing::ValuesIn(std::vector<size_t> {2, 4}), // seq lengths
            ::testing::ValuesIn(std::vector<size_t> {1}), // only single batch supported
            ::testing::ValuesIn(std::vector<size_t> {2, 4}), // hidden size
            ::testing::ValuesIn(std::vector<size_t> {0, 1}), // seq axis
            ::testing::ValuesIn(std::vector<float> {0.f}), // clip - not used
            ::testing::ValuesIn(std::vector<ngraph::helpers::TensorIteratorBody> {
                ngraph::helpers::TensorIteratorBody::LSTM,
                ngraph::helpers::TensorIteratorBody::RNN,
                ngraph::helpers::TensorIteratorBody::GRU,
            }), // body type
            ::testing::ValuesIn(std::vector<ngraph::op::RecurrentSequenceDirection>{
                ngraph::op::RecurrentSequenceDirection::FORWARD,
                ngraph::op::RecurrentSequenceDirection::REVERSE,
            }),
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision> {
                InferenceEngine::Precision::FP32,
                InferenceEngine::Precision::FP16,
            }), // precision
            ::testing::ValuesIn(std::vector<Config> {
                {CommonTestUtils::DEVICE_GPU, {{GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, PluginConfigParams::YES}}},
                {CommonTestUtils::DEVICE_GPU, {{GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, PluginConfigParams::NO}}}
            })), // configuration
        TensorIteratorWithConfigTest::getTestCaseName);
}  // namespace
