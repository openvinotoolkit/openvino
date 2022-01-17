// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include "openvino/op/tensor_iterator.hpp"
#include "base_reference_test.hpp"
#include <ngraph/op/util/attr_types.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph/runtime/reference/sequences.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct TensorIteratorParams {
    template <class T>
    TensorIteratorParams(
        const size_t batchSize, const size_t inputSize, const size_t hiddenSize, const size_t seqLength,
        const float clip, const ngraph::op::RecurrentSequenceDirection& direction,
        const ngraph::helpers::TensorIteratorBody& body_type,
        const element::Type_t& iType,
        const std::vector<T>& XValues, const std::vector<T>& H_tValues,  const std::vector<T>& C_tValues, const std::vector<int64_t>& S_tValues,
        const std::vector<T>& WValues, const std::vector<T>& RValues, const std::vector<T>& BValues,
        const std::vector<T>& YValues, const std::vector<T>& HoValues, const std::vector<T>& CoValues,
        const std::string& testcaseName = "") :
        batchSize(batchSize), inputSize(inputSize), hiddenSize(hiddenSize), seqLength(seqLength),
        clip(clip), body_type(body_type), direction(direction), iType(iType), oType(iType),
        testcaseName(testcaseName) {
            numDirections = (direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

            Shape XShape = Shape{batchSize, seqLength, inputSize};
            Shape H_tShape = Shape{batchSize, hiddenSize};
            Shape C_tShape = Shape{batchSize, hiddenSize};
            Shape S_tShape = Shape{batchSize};
            Shape WShape = Shape{4 * hiddenSize, inputSize};
            Shape RShape = Shape{4 * hiddenSize, hiddenSize};
            Shape BShape = Shape{4 * hiddenSize};
            Shape YShape = Shape{batchSize, seqLength, hiddenSize};
            Shape HoShape = Shape{batchSize, hiddenSize};
            Shape CoShape = Shape{batchSize, hiddenSize};

            X = Tensor(XShape, iType, XValues);
            H_t = Tensor(H_tShape, iType, H_tValues);
            C_t = Tensor(C_tShape, iType, C_tValues);
            S_t = Tensor(S_tShape, element::Type_t::i64, S_tValues);
            W = Tensor(WShape, iType, WValues);
            R = Tensor(RShape, iType, RValues);
            B = Tensor(BShape, iType, BValues);
            Y = Tensor(YShape, oType, YValues);
            Ho = Tensor(HoShape, oType, HoValues);
            Co = Tensor(CoShape, oType, CoValues);
        }

    size_t batchSize;
    size_t inputSize;
    size_t hiddenSize;
    size_t seqLength;
    size_t numDirections;
    size_t sequenceAxis = 1;
    float clip;
    ngraph::helpers::TensorIteratorBody body_type;
    ngraph::op::RecurrentSequenceDirection direction;

    element::Type_t iType;
    element::Type_t oType;

    Tensor X;
    Tensor H_t;
    Tensor C_t;
    Tensor S_t;
    Tensor W;
    Tensor R;
    Tensor B;
    Tensor Y;
    Tensor Ho;
    Tensor Co;
    std::string testcaseName;
};

class ReferenceTensorIteratorTest : public testing::TestWithParam<TensorIteratorParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.C_t.data};
        refOutData = {params.Y.data, params.Ho.data, params.Co.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bSize=" << param.batchSize;
        result << "_iSize=" << param.inputSize;
        result << "_hSize=" << param.hiddenSize;
        result << "_sLength=" << param.seqLength;
        result << "_clip=" << param.clip;
        result << "_xType=" << param.X.type;
        result << "_xShape=" << param.X.shape;
        if (param.testcaseName != "") {
            result << "_direction=" << param.direction;
            result << "_=" << param.testcaseName;
        } else {
            result << "_direction=" << param.direction;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const TensorIteratorParams& params) {
        std::vector<std::vector<size_t>> inputShapes;
        std::shared_ptr<ov::Model> function;
        auto tensor_iterator = std::make_shared<ngraph::opset5::TensorIterator>();

        // Each case consist of 3 steps:
        // 1. Create TensorIterator body.
        // 2. Set PortMap
        // 3. Create outer function
        auto axis = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1},
                                                               std::vector<int64_t>{static_cast<int64_t>(params.sequenceAxis)});
        switch (params.body_type) {
            case ngraph::helpers::TensorIteratorBody::LSTM: {
                inputShapes = {
                        {{params.batchSize, params.seqLength, params.inputSize}, {params.batchSize, params.hiddenSize}, {params.batchSize, params.hiddenSize},
                        {4 * params.hiddenSize, params.inputSize}, {4 * params.hiddenSize, params.inputSize}, {4 * params.inputSize}},
                };
                if (params.sequenceAxis == 0) {
                    // swap params.batchSize and params.seqLength
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(params.iType, {inputShapes[0], inputShapes[1], inputShapes[2]});

                // 1. Create TensorIterator body.
                inputShapes[0][params.sequenceAxis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(params.iType, {inputShapes[0], inputShapes[1], inputShapes[2]});
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1], body_params[2]};

                auto W = std::make_shared<ngraph::opset1::Constant>(params.W.type, params.W.shape, params.W.data.data());
                auto R = std::make_shared<ngraph::opset1::Constant>(params.R.type, params.R.shape, params.R.data.data());
                auto B = std::make_shared<ngraph::opset1::Constant>(params.B.type, params.B.shape, params.B.data.data());
                auto lstm_cell = std::make_shared<ngraph::opset4::LSTMCell>(out_vector[0],
                                                                            out_vector[1],
                                                                            out_vector[2],
                                                                            W,
                                                                            R,
                                                                            B,
                                                                            params.hiddenSize,
                                                                            std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                                                            std::vector<float>{},
                                                                            std::vector<float>{},
                                                                            params.clip);

                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(lstm_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(unsqueeze),
                                             std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                             std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "lstm_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (params.direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, params.sequenceAxis);
                    tensor_iterator->get_concatenated_slices(results[0], 0, 1, 1, -1, params.sequenceAxis);
                } else if (params.direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, params.sequenceAxis);
                    tensor_iterator->get_concatenated_slices(results[0], -1, -1, 1, 0, params.sequenceAxis);
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
/*            case ngraph::helpers::TensorIteratorBody::GRU: {
                inputShapes = {
                        {{batch, seq_lengths, input_size}, {batch, hidden_size}, {3 * hidden_size, input_size},
                                {3 * hidden_size, hidden_size}, {3 * hidden_size}},
                };
                if (params.sequenceAxis == 0) {
                    // swap params.batchSize and params.seqLength
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(params.iType, {inputShapes[0], inputShapes[1]});

                // 1. Create TensorIterator body.
                inputShapes[0][params.sequenceAxis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(params.iType, {inputShapes[0], inputShapes[1]});
                std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1]};
                auto gru_cell = ngraph::builder::makeGRU(out_vector, WRB, params.hiddenSize, {"sigmoid", "tanh"},
                                                         {}, {}, params.clip, false);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(gru_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_cell->output(0)),
                                             std::make_shared<ngraph::opset1::Result>(unsqueeze)};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "gru_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (params.direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, params.sequenceAxis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, params.sequenceAxis);
                } else if (params.direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, params.sequenceAxis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, params.sequenceAxis);
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
                if (params.sequenceAxis == 0) {
                    // swap params.batchSize and params.seqLength
                    std::swap(inputShapes[0][0], inputShapes[0][1]);
                }
                auto outer_params = ngraph::builder::makeParams(params.iType, {inputShapes[0], inputShapes[1]});

                // 1. Create TensorIterator body.
                inputShapes[0][params.sequenceAxis] = 1; // sliced dimension
                auto body_params = ngraph::builder::makeParams(params.iType, {inputShapes[0], inputShapes[1]});
                std::vector<ngraph::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
                auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(body_params[0], axis);
                ngraph::OutputVector out_vector = {squeeze, body_params[1]};
                auto rnn_cell = ngraph::builder::makeRNN(out_vector, WRB, params.hiddenSize, {"tanh"}, {}, {}, params.clip);
                auto unsqueeze = std::make_shared<ngraph::opset5::Unsqueeze>(rnn_cell->output(0), axis);
                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(rnn_cell),
                                             std::make_shared<ngraph::opset1::Result>(unsqueeze)};
                auto body = std::make_shared<ngraph::Function>(results, body_params, "rnn_cell");
                tensor_iterator->set_function(body);

                // 2. Set PortMap
                if (params.direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, params.sequenceAxis);
                    tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, params.sequenceAxis);
                } else if (params.direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
                    tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, params.sequenceAxis);
                    tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, params.sequenceAxis);
                } else {
                    NGRAPH_CHECK(false, "Bidirectional case is not supported.");
                }

                tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
                tensor_iterator->get_iter_value(results[0]);

                // 3. Outer function
                function = std::make_shared<ngraph::Function>(ngraph::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)}, outer_params);
                break;
            } */
        }
        return function;
    }
};

TEST_P(ReferenceTensorIteratorTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<TensorIteratorParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<TensorIteratorParams> params {
        TensorIteratorParams(
            5, 10, 10, 10,
            0.7f, op::RecurrentSequenceDirection::FORWARD, ngraph::helpers::TensorIteratorBody::LSTM,
            ET,
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
                5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
                3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
                8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
                2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
                1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
                2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
                2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
                1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
                4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
                2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
                7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
                5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
                2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
                6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
                4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
                4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
                9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
                8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
                6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
                7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
                1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
                1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
                1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
                3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
                2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
                1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
                1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
                6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
                5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
                5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
                8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
                5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 9.13242,
                2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457, 5.55154, 7.24966, 5.1128,
                4.25147, 8.34407, 1.4123, 4.49045, 5.12671, 7.62159, 9.18673, 3.49665, 8.35992, 6.90684,
                1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283, 9.88249, 4.48422,
                8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155, 9.94928, 6.88321, 1.33685,
                7.4718, 7.19305, 6.47932, 1.9559, 3.52616, 7.98593, 9.0115, 5.59539, 7.44137, 1.70001,
                6.53774, 8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153,
                5.4073, 9.00437, 8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473,
                7.95084, 2.11852, 9.28916, 1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549,
                3.75218, 8.98228, 8.98158, 5.63695, 3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035,
                8.37644, 8.54954, 5.70516, 2.45744, 9.54079, 1.53504, 8.9785, 6.1691, 4.40962, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
                5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
                3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
                8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
                2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
                1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
                2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
                2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
                1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
                4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
                2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
                7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
                5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
                2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
                6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
                4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
                4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
                9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
                8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
                6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
                7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
                1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
                1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
                1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
                3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
                2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
                1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
                1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
                6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
                5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
                5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
                8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
                5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
                5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
                3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
                8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
                2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
                1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
                2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
                2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
                1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
                4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
                2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
                7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
                5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
                2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
                6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
                4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
                4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
                9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
                8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
                6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
                7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
                1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
                1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
                1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
                3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
                2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
                1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
                1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
                6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
                5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
                5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
                8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
                5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 10},
            std::vector<T>{
                0.528016, 0.668187, 0.668186, 0.635471, 0.668187, 0.659096, 0.666861, 0.666715, 0.668138, 0.668186,
                0.53964, 0.668141, 0.668109, 0.619255, 0.668141, 0.647193, 0.662341, 0.661921, 0.667534, 0.66811,
                0.54692, 0.667558, 0.667297, 0.604361, 0.667564, 0.631676, 0.652518, 0.651781, 0.664541, 0.667311,
                0.551576, 0.664629, 0.663703, 0.592106, 0.664652, 0.615579, 0.638092, 0.637163, 0.656733, 0.663751,
                0.554596, 0.656917, 0.655047, 0.582718, 0.656967, 0.601233, 0.621878, 0.620939, 0.643723, 0.65514,
                0.556574, 0.643984, 0.641397, 0.575854, 0.644055, 0.589658, 0.606642, 0.605821, 0.627796, 0.641522,
                0.557878, 0.628081, 0.625301, 0.570987, 0.628158, 0.580903, 0.593915, 0.593262, 0.611954, 0.625433,
                0.558742, 0.612216, 0.609684, 0.567605, 0.612287, 0.574556, 0.584071, 0.583581, 0.598219, 0.609803,
                0.559316, 0.598435, 0.596364, 0.565285, 0.598493, 0.57008, 0.576828, 0.576475, 0.587333, 0.596461,
                0.559698, 0.587499, 0.58592, 0.563707, 0.587544, 0.56698, 0.571671, 0.571423, 0.579197, 0.585993,
                0.668182, 0.66458, 0.667903, 0.667432, 0.658361, 0.667935, 0.668185, 0.667547, 0.667307, 0.668186,
                0.66803, 0.656815, 0.666091, 0.664171, 0.646084, 0.666251, 0.668096, 0.66459, 0.663738, 0.668113,
                0.666772, 0.643839, 0.66026, 0.655973, 0.630413, 0.660667, 0.667203, 0.656835, 0.655116, 0.667328,
                0.662084, 0.627922, 0.649014, 0.642661, 0.614386, 0.649671, 0.663395, 0.643868, 0.64149, 0.663807,
                0.652065, 0.61207, 0.633798, 0.626647, 0.600233, 0.634582, 0.654454, 0.627954, 0.625399, 0.65525,
                0.637519, 0.598314, 0.617618, 0.610903, 0.588883, 0.618381, 0.640604, 0.612099, 0.609772, 0.641672,
                0.621298, 0.587406, 0.602959, 0.597357, 0.580333, 0.603611, 0.624467, 0.598338, 0.596436, 0.625592,
                0.606134, 0.57925, 0.591004, 0.586675, 0.57415, 0.591515, 0.608935, 0.587425, 0.585974, 0.609946,
                0.593511, 0.573381, 0.581898, 0.578717, 0.569797, 0.582278, 0.595758, 0.579264, 0.578207, 0.596577,
                0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546, 0.57339, 0.572642, 0.586082,
                0.668174, 0.668159, 0.668178, 0.618792, 0.66788, 0.668183, 0.66818, 0.66818, 0.662345, 0.595566,
                0.667915, 0.667737, 0.667963, 0.603963, 0.665981, 0.668052, 0.668006, 0.668007, 0.652525, 0.585315,
                0.66615, 0.665341, 0.6664, 0.591792, 0.659985, 0.666907, 0.666636, 0.66664, 0.638101, 0.577728,
                0.660409, 0.658471, 0.661057, 0.582484, 0.648575, 0.662479, 0.661698, 0.661709, 0.621887, 0.572305,
                0.649254, 0.646247, 0.650314, 0.575687, 0.633281, 0.652764, 0.651396, 0.651414, 0.60665, 0.568515,
                0.634083, 0.630598, 0.635357, 0.57087, 0.617117, 0.638404, 0.636684, 0.636707, 0.593922, 0.565907,
                0.617895, 0.614559, 0.619142, 0.567524, 0.602533, 0.622196, 0.62046, 0.620482, 0.584076, 0.564129,
                0.603195, 0.600379, 0.604265, 0.56523, 0.59067, 0.606921, 0.605404, 0.605423, 0.576832, 0.562925,
                0.591189, 0.588995, 0.592029, 0.56367, 0.581651, 0.594139, 0.59293, 0.592946, 0.571674, 0.562114,
                0.582036, 0.580415, 0.582661, 0.562616, 0.57509, 0.584239, 0.583333, 0.583345, 0.568079, 0.561569,
                0.668139, 0.668063, 0.668139, 0.667082, 0.653793, 0.663397, 0.640434, 0.668175, 0.667092, 0.571849,
                0.667538, 0.666978, 0.667544, 0.663011, 0.639734, 0.654459, 0.624289, 0.667925, 0.663042, 0.5682,
                0.664556, 0.66269, 0.664578, 0.653734, 0.623561, 0.640611, 0.608777, 0.666203, 0.653791, 0.565691,
                0.656765, 0.653146, 0.65681, 0.639656, 0.608128, 0.624474, 0.59563, 0.660545, 0.639731, 0.563983,
                0.643768, 0.638894, 0.643833, 0.62348, 0.595107, 0.608942, 0.585363, 0.649473, 0.623558, 0.562827,
                0.627845, 0.622696, 0.627915, 0.608056, 0.584968, 0.595763, 0.577763, 0.634345, 0.608125, 0.562048,
                0.611999, 0.607362, 0.612063, 0.595049, 0.577477, 0.585464, 0.572329, 0.61815, 0.595104, 0.561524,
                0.598256, 0.594491, 0.598309, 0.584924, 0.572127, 0.577836, 0.568532, 0.603413, 0.584966, 0.561173,
                0.587362, 0.584504, 0.587403, 0.577445, 0.568392, 0.572381, 0.565918, 0.591359, 0.577475, 0.560938,
                0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568, 0.564137, 0.582163, 0.572127, 0.560781,
                0.668102, 0.668132, 0.66388, 0.667456, 0.657447, 0.606385, 0.667634, 0.620685, 0.668185, 0.668187,
                0.667244, 0.667485, 0.655394, 0.664256, 0.644744, 0.59371, 0.664921, 0.6056, 0.668088, 0.668142,
                0.663529, 0.664358, 0.641868, 0.656146, 0.628916, 0.583917, 0.65754, 0.593086, 0.667146, 0.667567,
                0.654712, 0.656356, 0.625799, 0.642901, 0.612988, 0.576717, 0.644878, 0.583449, 0.66321, 0.664664,
                0.640947, 0.643193, 0.610134, 0.626905, 0.599072, 0.571593, 0.629065, 0.57638, 0.654104, 0.656992,
                0.624826, 0.62722, 0.59673, 0.611138, 0.587988, 0.568023, 0.613126, 0.571356, 0.640142, 0.644091,
                0.609258, 0.611426, 0.586197, 0.59755, 0.579676, 0.56557, 0.599186, 0.567859, 0.623984, 0.628198,
                0.596018, 0.597785, 0.578369, 0.586822, 0.573683, 0.563901, 0.588076, 0.565458, 0.608505, 0.612324,
                0.585658, 0.587002, 0.572757, 0.578824, 0.569471, 0.562771, 0.57974, 0.563825, 0.59541, 0.598524,
                0.577977, 0.578955, 0.568828, 0.573079, 0.566562, 0.562011, 0.573728, 0.56272, 0.585197, 0.587567},
            std::vector<T>{
                0.559698, 0.587499, 0.58592, 0.563707, 0.587544, 0.56698, 0.571671, 0.571423, 0.579197, 0.585993,
                0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546, 0.57339, 0.572642, 0.586082,
                0.582036, 0.580415, 0.582661, 0.562616, 0.57509, 0.584239, 0.583333, 0.583345, 0.568079, 0.561569,
                0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568, 0.564137, 0.582163, 0.572127, 0.560781,
                0.577977, 0.578955, 0.568828, 0.573079, 0.566562, 0.562011, 0.573728, 0.56272, 0.585197, 0.587567},
            std::vector<T>{
                1.2132, 1.37242, 1.3621, 1.23365, 1.37271, 1.25089, 1.27652, 1.27513, 1.32014, 1.36258,
                1.34833, 1.26322, 1.29695, 1.284, 1.24985, 1.29853, 1.35913, 1.2862, 1.28197, 1.36315,
                1.33748, 1.32752, 1.34137, 1.22801, 1.29593, 1.35132, 1.34559, 1.34566, 1.25679, 1.22266,
                1.32026, 1.30789, 1.32044, 1.27895, 1.24474, 1.25944, 1.23589, 1.33827, 1.27907, 1.21865,
                1.31284, 1.31868, 1.26086, 1.28443, 1.24866, 1.22491, 1.28812, 1.22855, 1.35744, 1.37287}),
        TensorIteratorParams(
            5, 10, 10, 10,
            0.7f, op::RecurrentSequenceDirection::REVERSE, ngraph::helpers::TensorIteratorBody::LSTM,
            ET,
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
                5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
                3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
                8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
                2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
                1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
                2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
                2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
                1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
                4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
                2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
                7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
                5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
                2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
                6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
                4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
                4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
                9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
                8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
                6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
                7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
                1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
                1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
                1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
                3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
                2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
                1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
                1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
                6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
                5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
                5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
                8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
                5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 9.13242,
                2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457, 5.55154, 7.24966, 5.1128,
                4.25147, 8.34407, 1.4123, 4.49045, 5.12671, 7.62159, 9.18673, 3.49665, 8.35992, 6.90684,
                1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283, 9.88249, 4.48422,
                8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155, 9.94928, 6.88321, 1.33685,
                7.4718, 7.19305, 6.47932, 1.9559, 3.52616, 7.98593, 9.0115, 5.59539, 7.44137, 1.70001,
                6.53774, 8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153,
                5.4073, 9.00437, 8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473,
                7.95084, 2.11852, 9.28916, 1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549,
                3.75218, 8.98228, 8.98158, 5.63695, 3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035,
                8.37644, 8.54954, 5.70516, 2.45744, 9.54079, 1.53504, 8.9785, 6.1691, 4.40962, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
                5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
                3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
                8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
                2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
                1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
                2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
                2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
                1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
                4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
                2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
                7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
                5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
                2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
                6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
                4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
                4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
                9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
                8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
                6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
                7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
                1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
                1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
                1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
                3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
                2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
                1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
                1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
                6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
                5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
                5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
                8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
                5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
                5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
                3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
                8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
                2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
                1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
                2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
                2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
                1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
                4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
                2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
                7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
                5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
                2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
                6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
                4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
                4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
                9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
                8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
                6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
                7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
                1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
                1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
                1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
                3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
                2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
                1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
                1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
                6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
                5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
                5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
                8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
                5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{
                1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
                8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 10},
            std::vector<T>{
                0.559698, 0.587499, 0.58592, 0.563707, 0.587544, 0.56698, 0.571671, 0.571423, 0.579197, 0.585993,
                0.559316, 0.598435, 0.596364, 0.565285, 0.598493, 0.57008, 0.576828, 0.576475, 0.587333, 0.596461,
                0.558742, 0.612216, 0.609684, 0.567605, 0.612287, 0.574556, 0.584071, 0.583581, 0.598219, 0.609803,
                0.557878, 0.628081, 0.625301, 0.570987, 0.628158, 0.580903, 0.593915, 0.593262, 0.611954, 0.625433,
                0.556574, 0.643984, 0.641397, 0.575854, 0.644055, 0.589658, 0.606642, 0.605821, 0.627796, 0.641522,
                0.554596, 0.656917, 0.655047, 0.582718, 0.656967, 0.601233, 0.621878, 0.620939, 0.643723, 0.65514,
                0.551576, 0.664629, 0.663703, 0.592106, 0.664652, 0.615579, 0.638092, 0.637163, 0.656733, 0.663751,
                0.54692, 0.667558, 0.667297, 0.604361, 0.667564, 0.631676, 0.652518, 0.651781, 0.664541, 0.667311,
                0.53964, 0.668141, 0.668109, 0.619255, 0.668141, 0.647193, 0.662341, 0.661921, 0.667534, 0.66811,
                0.528016, 0.668187, 0.668186, 0.635471, 0.668187, 0.659096, 0.666861, 0.666715, 0.668138, 0.668186,
                0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546, 0.57339, 0.572642, 0.586082,
                0.593511, 0.573381, 0.581898, 0.578717, 0.569797, 0.582278, 0.595758, 0.579264, 0.578207, 0.596577,
                0.606134, 0.57925, 0.591004, 0.586675, 0.57415, 0.591515, 0.608935, 0.587425, 0.585974, 0.609946,
                0.621298, 0.587406, 0.602959, 0.597357, 0.580333, 0.603611, 0.624467, 0.598338, 0.596436, 0.625592,
                0.637519, 0.598314, 0.617618, 0.610903, 0.588883, 0.618381, 0.640604, 0.612099, 0.609772, 0.641672,
                0.652065, 0.61207, 0.633798, 0.626647, 0.600233, 0.634582, 0.654454, 0.627954, 0.625399, 0.65525,
                0.662084, 0.627922, 0.649014, 0.642661, 0.614386, 0.649671, 0.663395, 0.643868, 0.64149, 0.663807,
                0.666772, 0.643839, 0.66026, 0.655973, 0.630413, 0.660667, 0.667203, 0.656835, 0.655116, 0.667328,
                0.66803, 0.656815, 0.666091, 0.664171, 0.646084, 0.666251, 0.668096, 0.66459, 0.663738, 0.668113,
                0.668182, 0.66458, 0.667903, 0.667432, 0.658361, 0.667935, 0.668185, 0.667547, 0.667307, 0.668186,
                0.582036, 0.580415, 0.582661, 0.562616, 0.57509, 0.584239, 0.583333, 0.583345, 0.568079, 0.561569,
                0.591189, 0.588995, 0.592029, 0.56367, 0.581651, 0.594139, 0.59293, 0.592946, 0.571674, 0.562114,
                0.603195, 0.600379, 0.604265, 0.56523, 0.59067, 0.606921, 0.605404, 0.605423, 0.576832, 0.562925,
                0.617895, 0.614559, 0.619142, 0.567524, 0.602533, 0.622196, 0.62046, 0.620482, 0.584076, 0.564129,
                0.634083, 0.630598, 0.635357, 0.57087, 0.617117, 0.638404, 0.636684, 0.636707, 0.593922, 0.565907,
                0.649254, 0.646247, 0.650314, 0.575687, 0.633281, 0.652764, 0.651396, 0.651414, 0.60665, 0.568515,
                0.660409, 0.658471, 0.661057, 0.582484, 0.648575, 0.662479, 0.661698, 0.661709, 0.621887, 0.572305,
                0.66615, 0.665341, 0.6664, 0.591792, 0.659985, 0.666907, 0.666636, 0.66664, 0.638101, 0.577728,
                0.667915, 0.667737, 0.667963, 0.603963, 0.665981, 0.668052, 0.668006, 0.668007, 0.652525, 0.585315,
                0.668174, 0.668159, 0.668178, 0.618792, 0.66788, 0.668183, 0.66818, 0.66818, 0.662345, 0.595566,
                0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568, 0.564137, 0.582163, 0.572127, 0.560781,
                0.587362, 0.584504, 0.587403, 0.577445, 0.568392, 0.572381, 0.565918, 0.591359, 0.577475, 0.560938,
                0.598256, 0.594491, 0.598309, 0.584924, 0.572127, 0.577836, 0.568532, 0.603413, 0.584966, 0.561173,
                0.611999, 0.607362, 0.612063, 0.595049, 0.577477, 0.585464, 0.572329, 0.61815, 0.595104, 0.561524,
                0.627845, 0.622696, 0.627915, 0.608056, 0.584968, 0.595763, 0.577763, 0.634345, 0.608125, 0.562048,
                0.643768, 0.638894, 0.643833, 0.62348, 0.595107, 0.608942, 0.585363, 0.649473, 0.623558, 0.562827,
                0.656765, 0.653146, 0.65681, 0.639656, 0.608128, 0.624474, 0.59563, 0.660545, 0.639731, 0.563983,
                0.664556, 0.66269, 0.664578, 0.653734, 0.623561, 0.640611, 0.608777, 0.666203, 0.653791, 0.565691,
                0.667538, 0.666978, 0.667544, 0.663011, 0.639734, 0.654459, 0.624289, 0.667925, 0.663042, 0.5682,
                0.668139, 0.668063, 0.668139, 0.667082, 0.653793, 0.663397, 0.640434, 0.668175, 0.667092, 0.571849,
                0.577977, 0.578955, 0.568828, 0.573079, 0.566562, 0.562011, 0.573728, 0.56272, 0.585197, 0.587567,
                0.585658, 0.587002, 0.572757, 0.578824, 0.569471, 0.562771, 0.57974, 0.563825, 0.59541, 0.598524,
                0.596018, 0.597785, 0.578369, 0.586822, 0.573683, 0.563901, 0.588076, 0.565458, 0.608505, 0.612324,
                0.609258, 0.611426, 0.586197, 0.59755, 0.579676, 0.56557, 0.599186, 0.567859, 0.623984, 0.628198,
                0.624826, 0.62722, 0.59673, 0.611138, 0.587988, 0.568023, 0.613126, 0.571356, 0.640142, 0.644091,
                0.640947, 0.643193, 0.610134, 0.626905, 0.599072, 0.571593, 0.629065, 0.57638, 0.654104, 0.656992,
                0.654712, 0.656356, 0.625799, 0.642901, 0.612988, 0.576717, 0.644878, 0.583449, 0.66321, 0.664664,
                0.663529, 0.664358, 0.641868, 0.656146, 0.628916, 0.583917, 0.65754, 0.593086, 0.667146, 0.667567,
                0.667244, 0.667485, 0.655394, 0.664256, 0.644744, 0.59371, 0.664921, 0.6056, 0.668088, 0.668142,
                0.668102, 0.668132, 0.66388, 0.667456, 0.657447, 0.606385, 0.667634, 0.620685, 0.668185, 0.668187},
            std::vector<T>{
                0.559698, 0.587499, 0.58592, 0.563707, 0.587544, 0.56698, 0.571671, 0.571423, 0.579197, 0.585993,
                0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546, 0.57339, 0.572642, 0.586082,
                0.582036, 0.580415, 0.582661, 0.562616, 0.57509, 0.584239, 0.583333, 0.583345, 0.568079, 0.561569,
                0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568, 0.564137, 0.582163, 0.572127, 0.560781,
                0.577977, 0.578955, 0.568828, 0.573079, 0.566562, 0.562011, 0.573728, 0.56272, 0.585197, 0.587567},
            std::vector<T>{
                1.2132, 1.37242, 1.3621, 1.23365, 1.37271, 1.25089, 1.27652, 1.27513, 1.32014, 1.36258,
                1.34833, 1.26322, 1.29695, 1.284, 1.24985, 1.29853, 1.35913, 1.2862, 1.28197, 1.36315,
                1.33748, 1.32752, 1.34137, 1.22801, 1.29593, 1.35132, 1.34559, 1.34566, 1.25679, 1.22266,
                1.32026, 1.30789, 1.32044, 1.27895, 1.24474, 1.25944, 1.23589, 1.33827, 1.27907, 1.21865,
                1.31284, 1.31868, 1.26086, 1.28443, 1.24866, 1.22491, 1.28812, 1.22855, 1.35744, 1.37287}),
    };
    return params;
}

template <element::Type_t ET>
std::vector<TensorIteratorParams> generateParamsBF16() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<TensorIteratorParams> params {
        TensorIteratorParams(
            5, 10, 10, 10,
            0.7f, op::RecurrentSequenceDirection::FORWARD, ngraph::helpers::TensorIteratorBody::LSTM,
            ET,
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 8.875,
                8.75, 9, 8.4375, 1.76562, 8.4375, 1.34375, 3.45312, 2.53125, 1.53125, 8.875,
                7.03125, 1.88281, 6.3125, 4.78125, 7.03125, 9.625, 4.6875, 5.8125, 2.78125, 7.21875,
                3.59375, 3.84375, 2.28125, 7.1875, 8, 8.5, 4.6875, 1.16406, 1.30469, 7.75,
                6.625, 9.875, 6.9375, 7.71875, 3.6875, 3.53125, 5, 8.125, 3, 1.92188,
                1.65625, 5, 5.21875, 9.125, 1.85938, 3.64062, 9.125, 3.59375, 2.0625, 2.15625,
                5.71875, 1.17188, 1.75, 7.125, 9.25, 2.90625, 9.1875, 3.375, 3.6875, 5.4375,
                6.25, 1.47656, 6.0625, 6.15625, 6.5, 2.3125, 9.625, 6.3125, 3.34375, 7.3125,
                3.07812, 1.92188, 5.8125, 4.71875, 9.5, 7.25, 5.4375, 4.71875, 5.875, 1.45312,
                7.875, 5.8125, 1.40625, 6.96875, 2.25, 5.625, 8.125, 9.5, 1.26562, 6.25,
                8.9375, 9.125, 5.875, 2.23438, 5.03125, 2.25, 9, 8.25, 4.375, 4.5625,
                5.84375, 2.48438, 6.875, 9.375, 4.25, 4.125, 6.125, 7.75, 6.75, 7.53125,
                2.125, 8.9375, 7.1875, 6.625, 6.8125, 7.75, 4.1875, 4.125, 7.875, 3.42188,
                4.1875, 9.0625, 7.75, 4.84375, 8.875, 9.625, 1.10156, 6.96875, 5.46875, 6.59375,
                1.66406, 2.03125, 8.0625, 9.5, 1.57812, 5.0625, 4.1875, 6.1875, 9.5, 4.6875,
                4.40625, 3.125, 7.875, 9.125, 7.9375, 6.15625, 3.71875, 1.02344, 7.9375, 6.5625,
                2.375, 3.9375, 6.1875, 5.75, 1.07812, 9, 7.375, 4.1875, 5.25, 9.125,
                7.875, 6.625, 5.1875, 1.14062, 3.40625, 9.375, 8.5, 7.1875, 5.9375, 10,
                1.625, 2.54688, 5.25, 2.21875, 7.6875, 9.375, 2.71875, 7.25, 5.1875, 1.59375,
                3.0625, 7.8125, 5.5625, 7.78125, 2.875, 9.25, 1.4375, 7.375, 5.65625, 2.125,
                2.54688, 1.17188, 4.5625, 1.23438, 1.96875, 1.25, 5.5625, 3.21875, 1.92188, 8.75,
                3.59375, 5.84375, 3.07812, 5.96875, 9.6875, 8.5625, 3.5, 2.125, 3.09375, 3.5,
                1.82031, 6.25, 6.125, 9.75, 4.75, 6.0625, 4.3125, 1.16406, 8.3125, 8.1875,
                3.59375, 3.09375, 7.4375, 8.25, 6.5, 4.5, 4.8125, 8.75, 7.75, 7.71875,
                4.84375, 6, 4.84375, 2.21875, 4.25, 1.53906, 2.375, 2.09375, 9.375, 1.39844,
                9.25, 1.96875, 8, 3.03125, 6.5625, 7.40625, 1.32031, 6.03125, 6.875, 1.10938,
                2.15625, 1.64062, 3.65625, 9.6875, 4.25, 6.125, 3.46875, 2.82812, 1.66406, 3.26562,
                2.375, 7.6875, 2.45312, 2.75, 9.4375, 6.21875, 4.3125, 9.75, 1.45312, 8.625,
                7.65625, 3.15625, 3.6875, 5.4375, 2.84375, 6.5625, 9.8125, 8.4375, 9, 2.40625,
                7.8125, 1.16406, 6.875, 1.625, 1.35938, 5.375, 8.3125, 6.4375, 7.875, 6.125,
                5.09375, 3.84375, 5.78125, 9.875, 1.98438, 6.1875, 2.3125, 4.40625, 5.5625, 5.9375,
                2.9375, 7.6875, 9.25, 7, 5.15625, 3.375, 2.1875, 1.59375, 7.875, 4.3125,
                2.90625, 6.65625, 1.67188, 2.89062, 1.85938, 7.75, 2.45312, 1.59375, 4.1875, 3.34375,
                1.85938, 8.25, 2.28125, 2.73438, 9.375, 6.75, 6.1875, 5.71875, 8.5, 9.3125,
                6.625, 3.375, 3.90625, 1.59375, 7.5625, 7.625, 5.6875, 7.9375, 7.625, 9.125,
                2.48438, 9.375, 7.1875, 1.125, 4.8125, 3.09375, 7.5625, 6.5625, 7.8125, 9.5,
                4.5625, 9.5, 9.3125, 6, 2.82812, 9.25, 1.07031, 6.75, 9.3125, 4.5,
                3.65625, 5.375, 2.5, 6.4375, 1.21875, 5.9375, 5.0625, 9.3125, 8.25, 9.25,
                4.3125, 4.5625, 6.46875, 9.625, 1.3125, 2.5625, 4.1875, 2.125, 1.70312, 2.21875,
                7.25, 5.5625, 1.10938, 1.1875, 5.125, 9.5, 9.625, 8.4375, 4, 1.13281,
                5.25, 2.57812, 1.94531, 3.98438, 5.5, 2.17188, 9, 8.25, 5.8125, 4.09375,
                3.53125, 9.4375, 4.1875, 6.25, 9.0625, 8.875, 3.17188, 8.625, 1.21875, 9.125,
                9.6875, 5.125, 4.875, 5.90625, 4.125, 8.125, 6.1875, 3.5625, 2.125, 5.40625,
                9.5, 6.375, 3.8125, 1.14062, 9.5625, 6.3125, 2.96875, 4.875, 3.23438, 8.25,
                8.75, 3.84375, 3.125, 9, 8.3125, 6.1875, 5.875, 2.65625, 2.71875, 8.0625,
                6.3125, 6.5, 1.42969, 1.48438, 1.14062, 4.78125, 1.44531, 7.125, 4.59375, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 8.875,
                8.75, 9, 8.4375, 1.76562, 8.4375, 1.34375, 3.45312, 2.53125, 1.53125, 8.875,
                7.03125, 1.88281, 6.3125, 4.78125, 7.03125, 9.625, 4.6875, 5.8125, 2.78125, 7.21875,
                3.59375, 3.84375, 2.28125, 7.1875, 8, 8.5, 4.6875, 1.16406, 1.30469, 7.75,
                6.625, 9.875, 6.9375, 7.71875, 3.6875, 3.53125, 5, 8.125, 3, 1.92188,
                1.65625, 5, 5.21875, 9.125, 1.85938, 3.64062, 9.125, 3.59375, 2.0625, 2.15625,
                5.71875, 1.17188, 1.75, 7.125, 9.25, 2.90625, 9.1875, 3.375, 3.6875, 5.4375,
                6.25, 1.47656, 6.0625, 6.15625, 6.5, 2.3125, 9.625, 6.3125, 3.34375, 7.3125,
                3.07812, 1.92188, 5.8125, 4.71875, 9.5, 7.25, 5.4375, 4.71875, 5.875, 1.45312,
                7.875, 5.8125, 1.40625, 6.96875, 2.25, 5.625, 8.125, 9.5, 1.26562, 6.25,
                8.9375, 9.125, 5.875, 2.23438, 5.03125, 2.25, 9, 8.25, 4.375, 4.5625,
                5.84375, 2.48438, 6.875, 9.375, 4.25, 4.125, 6.125, 7.75, 6.75, 7.53125,
                2.125, 8.9375, 7.1875, 6.625, 6.8125, 7.75, 4.1875, 4.125, 7.875, 3.42188,
                4.1875, 9.0625, 7.75, 4.84375, 8.875, 9.625, 1.10156, 6.96875, 5.46875, 6.59375,
                1.66406, 2.03125, 8.0625, 9.5, 1.57812, 5.0625, 4.1875, 6.1875, 9.5, 4.6875,
                4.40625, 3.125, 7.875, 9.125, 7.9375, 6.15625, 3.71875, 1.02344, 7.9375, 6.5625,
                2.375, 3.9375, 6.1875, 5.75, 1.07812, 9, 7.375, 4.1875, 5.25, 9.125,
                7.875, 6.625, 5.1875, 1.14062, 3.40625, 9.375, 8.5, 7.1875, 5.9375, 10,
                1.625, 2.54688, 5.25, 2.21875, 7.6875, 9.375, 2.71875, 7.25, 5.1875, 1.59375,
                3.0625, 7.8125, 5.5625, 7.78125, 2.875, 9.25, 1.4375, 7.375, 5.65625, 2.125,
                2.54688, 1.17188, 4.5625, 1.23438, 1.96875, 1.25, 5.5625, 3.21875, 1.92188, 8.75,
                3.59375, 5.84375, 3.07812, 5.96875, 9.6875, 8.5625, 3.5, 2.125, 3.09375, 3.5,
                1.82031, 6.25, 6.125, 9.75, 4.75, 6.0625, 4.3125, 1.16406, 8.3125, 8.1875,
                3.59375, 3.09375, 7.4375, 8.25, 6.5, 4.5, 4.8125, 8.75, 7.75, 7.71875,
                4.84375, 6, 4.84375, 2.21875, 4.25, 1.53906, 2.375, 2.09375, 9.375, 1.39844,
                9.25, 1.96875, 8, 3.03125, 6.5625, 7.40625, 1.32031, 6.03125, 6.875, 1.10938,
                2.15625, 1.64062, 3.65625, 9.6875, 4.25, 6.125, 3.46875, 2.82812, 1.66406, 3.26562,
                2.375, 7.6875, 2.45312, 2.75, 9.4375, 6.21875, 4.3125, 9.75, 1.45312, 8.625,
                7.65625, 3.15625, 3.6875, 5.4375, 2.84375, 6.5625, 9.8125, 8.4375, 9, 2.40625,
                7.8125, 1.16406, 6.875, 1.625, 1.35938, 5.375, 8.3125, 6.4375, 7.875, 6.125,
                5.09375, 3.84375, 5.78125, 9.875, 1.98438, 6.1875, 2.3125, 4.40625, 5.5625, 5.9375,
                2.9375, 7.6875, 9.25, 7, 5.15625, 3.375, 2.1875, 1.59375, 7.875, 4.3125,
                2.90625, 6.65625, 1.67188, 2.89062, 1.85938, 7.75, 2.45312, 1.59375, 4.1875, 3.34375,
                1.85938, 8.25, 2.28125, 2.73438, 9.375, 6.75, 6.1875, 5.71875, 8.5, 9.3125,
                6.625, 3.375, 3.90625, 1.59375, 7.5625, 7.625, 5.6875, 7.9375, 7.625, 9.125,
                2.48438, 9.375, 7.1875, 1.125, 4.8125, 3.09375, 7.5625, 6.5625, 7.8125, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 8.875,
                8.75, 9, 8.4375, 1.76562, 8.4375, 1.34375, 3.45312, 2.53125, 1.53125, 8.875,
                7.03125, 1.88281, 6.3125, 4.78125, 7.03125, 9.625, 4.6875, 5.8125, 2.78125, 7.21875,
                3.59375, 3.84375, 2.28125, 7.1875, 8, 8.5, 4.6875, 1.16406, 1.30469, 7.75,
                6.625, 9.875, 6.9375, 7.71875, 3.6875, 3.53125, 5, 8.125, 3, 1.92188,
                1.65625, 5, 5.21875, 9.125, 1.85938, 3.64062, 9.125, 3.59375, 2.0625, 2.15625,
                5.71875, 1.17188, 1.75, 7.125, 9.25, 2.90625, 9.1875, 3.375, 3.6875, 5.4375,
                6.25, 1.47656, 6.0625, 6.15625, 6.5, 2.3125, 9.625, 6.3125, 3.34375, 7.3125,
                3.07812, 1.92188, 5.8125, 4.71875, 9.5, 7.25, 5.4375, 4.71875, 5.875, 1.45312,
                7.875, 5.8125, 1.40625, 6.96875, 2.25, 5.625, 8.125, 9.5, 1.26562, 6.25,
                8.9375, 9.125, 5.875, 2.23438, 5.03125, 2.25, 9, 8.25, 4.375, 4.5625,
                5.84375, 2.48438, 6.875, 9.375, 4.25, 4.125, 6.125, 7.75, 6.75, 7.53125,
                2.125, 8.9375, 7.1875, 6.625, 6.8125, 7.75, 4.1875, 4.125, 7.875, 3.42188,
                4.1875, 9.0625, 7.75, 4.84375, 8.875, 9.625, 1.10156, 6.96875, 5.46875, 6.59375,
                1.66406, 2.03125, 8.0625, 9.5, 1.57812, 5.0625, 4.1875, 6.1875, 9.5, 4.6875,
                4.40625, 3.125, 7.875, 9.125, 7.9375, 6.15625, 3.71875, 1.02344, 7.9375, 6.5625,
                2.375, 3.9375, 6.1875, 5.75, 1.07812, 9, 7.375, 4.1875, 5.25, 9.125,
                7.875, 6.625, 5.1875, 1.14062, 3.40625, 9.375, 8.5, 7.1875, 5.9375, 10,
                1.625, 2.54688, 5.25, 2.21875, 7.6875, 9.375, 2.71875, 7.25, 5.1875, 1.59375,
                3.0625, 7.8125, 5.5625, 7.78125, 2.875, 9.25, 1.4375, 7.375, 5.65625, 2.125,
                2.54688, 1.17188, 4.5625, 1.23438, 1.96875, 1.25, 5.5625, 3.21875, 1.92188, 8.75,
                3.59375, 5.84375, 3.07812, 5.96875, 9.6875, 8.5625, 3.5, 2.125, 3.09375, 3.5,
                1.82031, 6.25, 6.125, 9.75, 4.75, 6.0625, 4.3125, 1.16406, 8.3125, 8.1875,
                3.59375, 3.09375, 7.4375, 8.25, 6.5, 4.5, 4.8125, 8.75, 7.75, 7.71875,
                4.84375, 6, 4.84375, 2.21875, 4.25, 1.53906, 2.375, 2.09375, 9.375, 1.39844,
                9.25, 1.96875, 8, 3.03125, 6.5625, 7.40625, 1.32031, 6.03125, 6.875, 1.10938,
                2.15625, 1.64062, 3.65625, 9.6875, 4.25, 6.125, 3.46875, 2.82812, 1.66406, 3.26562,
                2.375, 7.6875, 2.45312, 2.75, 9.4375, 6.21875, 4.3125, 9.75, 1.45312, 8.625,
                7.65625, 3.15625, 3.6875, 5.4375, 2.84375, 6.5625, 9.8125, 8.4375, 9, 2.40625,
                7.8125, 1.16406, 6.875, 1.625, 1.35938, 5.375, 8.3125, 6.4375, 7.875, 6.125,
                5.09375, 3.84375, 5.78125, 9.875, 1.98438, 6.1875, 2.3125, 4.40625, 5.5625, 5.9375,
                2.9375, 7.6875, 9.25, 7, 5.15625, 3.375, 2.1875, 1.59375, 7.875, 4.3125,
                2.90625, 6.65625, 1.67188, 2.89062, 1.85938, 7.75, 2.45312, 1.59375, 4.1875, 3.34375,
                1.85938, 8.25, 2.28125, 2.73438, 9.375, 6.75, 6.1875, 5.71875, 8.5, 9.3125,
                6.625, 3.375, 3.90625, 1.59375, 7.5625, 7.625, 5.6875, 7.9375, 7.625, 9.125,
                2.48438, 9.375, 7.1875, 1.125, 4.8125, 3.09375, 7.5625, 6.5625, 7.8125, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 10},
            std::vector<T>{
                0.523438, 0.667969, 0.667969, 0.667969, 0.667969, 0.523438, 0.632812, 0.664062, 0.667969, 0.640625,
                0.539062, 0.664062, 0.667969, 0.667969, 0.667969, 0.539062, 0.617188, 0.65625, 0.667969, 0.625,
                0.546875, 0.648438, 0.667969, 0.664062, 0.667969, 0.546875, 0.601562, 0.640625, 0.667969, 0.609375,
                0.546875, 0.632812, 0.664062, 0.65625, 0.664062, 0.546875, 0.585938, 0.625, 0.664062, 0.59375,
                0.554688, 0.617188, 0.65625, 0.640625, 0.648438, 0.554688, 0.578125, 0.609375, 0.65625, 0.585938,
                0.554688, 0.601562, 0.640625, 0.625, 0.640625, 0.554688, 0.570312, 0.59375, 0.640625, 0.578125,
                0.554688, 0.59375, 0.625, 0.609375, 0.625, 0.554688, 0.570312, 0.585938, 0.625, 0.570312,
                0.554688, 0.585938, 0.609375, 0.59375, 0.609375, 0.554688, 0.5625, 0.578125, 0.609375, 0.570312,
                0.554688, 0.570312, 0.59375, 0.585938, 0.59375, 0.554688, 0.5625, 0.570312, 0.59375, 0.5625,
                0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625, 0.570312, 0.585938, 0.5625,
                0.65625, 0.617188, 0.664062, 0.648438, 0.664062, 0.664062, 0.667969, 0.664062, 0.667969, 0.667969,
                0.648438, 0.601562, 0.664062, 0.632812, 0.664062, 0.65625, 0.667969, 0.664062, 0.667969, 0.664062,
                0.632812, 0.585938, 0.648438, 0.617188, 0.648438, 0.648438, 0.664062, 0.648438, 0.667969, 0.65625,
                0.617188, 0.578125, 0.632812, 0.601562, 0.632812, 0.632812, 0.65625, 0.632812, 0.664062, 0.648438,
                0.601562, 0.570312, 0.617188, 0.585938, 0.617188, 0.617188, 0.640625, 0.617188, 0.648438, 0.632812,
                0.585938, 0.570312, 0.601562, 0.578125, 0.601562, 0.601562, 0.625, 0.601562, 0.640625, 0.617188,
                0.578125, 0.5625, 0.585938, 0.570312, 0.585938, 0.585938, 0.609375, 0.585938, 0.625, 0.601562,
                0.570312, 0.5625, 0.578125, 0.570312, 0.578125, 0.578125, 0.59375, 0.578125, 0.609375, 0.585938,
                0.570312, 0.5625, 0.570312, 0.5625, 0.570312, 0.570312, 0.585938, 0.570312, 0.59375, 0.578125,
                0.5625, 0.554688, 0.570312, 0.5625, 0.570312, 0.570312, 0.578125, 0.570312, 0.585938, 0.570312,
                0.667969, 0.667969, 0.664062, 0.667969, 0.667969, 0.648438, 0.667969, 0.667969, 0.65625, 0.5625,
                0.667969, 0.664062, 0.65625, 0.667969, 0.664062, 0.640625, 0.664062, 0.667969, 0.640625, 0.5625,
                0.664062, 0.648438, 0.640625, 0.664062, 0.65625, 0.625, 0.65625, 0.664062, 0.625, 0.554688,
                0.65625, 0.632812, 0.625, 0.65625, 0.648438, 0.609375, 0.640625, 0.664062, 0.609375, 0.554688,
                0.648438, 0.617188, 0.609375, 0.640625, 0.632812, 0.59375, 0.625, 0.648438, 0.59375, 0.554688,
                0.632812, 0.601562, 0.59375, 0.625, 0.617188, 0.585938, 0.609375, 0.632812, 0.585938, 0.554688,
                0.617188, 0.59375, 0.585938, 0.609375, 0.601562, 0.578125, 0.59375, 0.617188, 0.578125, 0.554688,
                0.601562, 0.585938, 0.578125, 0.59375, 0.585938, 0.570312, 0.585938, 0.601562, 0.570312, 0.554688,
                0.585938, 0.570312, 0.570312, 0.585938, 0.578125, 0.570312, 0.578125, 0.585938, 0.570312, 0.554688,
                0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625, 0.570312, 0.578125, 0.5625, 0.554688,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.632812, 0.667969, 0.648438,
                0.664062, 0.667969, 0.667969, 0.664062, 0.664062, 0.664062, 0.664062, 0.617188, 0.667969, 0.632812,
                0.65625, 0.664062, 0.667969, 0.648438, 0.65625, 0.65625, 0.648438, 0.601562, 0.667969, 0.617188,
                0.648438, 0.65625, 0.664062, 0.632812, 0.640625, 0.648438, 0.640625, 0.59375, 0.664062, 0.601562,
                0.632812, 0.640625, 0.648438, 0.617188, 0.625, 0.632812, 0.625, 0.585938, 0.648438, 0.59375,
                0.617188, 0.625, 0.632812, 0.601562, 0.609375, 0.617188, 0.609375, 0.570312, 0.640625, 0.585938,
                0.601562, 0.609375, 0.617188, 0.59375, 0.59375, 0.601562, 0.59375, 0.570312, 0.625, 0.570312,
                0.585938, 0.59375, 0.601562, 0.585938, 0.585938, 0.585938, 0.585938, 0.5625, 0.609375, 0.570312,
                0.578125, 0.585938, 0.59375, 0.570312, 0.578125, 0.578125, 0.578125, 0.5625, 0.59375, 0.5625,
                0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312, 0.570312, 0.5625, 0.585938, 0.5625,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.664062, 0.617188, 0.667969, 0.667969, 0.667969,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.65625, 0.601562, 0.667969, 0.664062, 0.667969,
                0.664062, 0.664062, 0.664062, 0.667969, 0.664062, 0.640625, 0.585938, 0.664062, 0.65625, 0.667969,
                0.65625, 0.65625, 0.65625, 0.664062, 0.65625, 0.625, 0.578125, 0.65625, 0.648438, 0.664062,
                0.648438, 0.648438, 0.640625, 0.65625, 0.648438, 0.609375, 0.570312, 0.640625, 0.632812, 0.65625,
                0.632812, 0.632812, 0.625, 0.640625, 0.632812, 0.59375, 0.570312, 0.625, 0.617188, 0.640625,
                0.617188, 0.617188, 0.609375, 0.625, 0.617188, 0.585938, 0.5625, 0.609375, 0.601562, 0.625,
                0.601562, 0.601562, 0.59375, 0.609375, 0.601562, 0.578125, 0.5625, 0.59375, 0.585938, 0.609375,
                0.585938, 0.585938, 0.585938, 0.59375, 0.585938, 0.570312, 0.5625, 0.585938, 0.578125, 0.59375,
                0.578125, 0.578125, 0.578125, 0.585938, 0.578125, 0.570312, 0.554688, 0.578125, 0.570312, 0.585938},
            std::vector<T>{
                0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625, 0.570312, 0.585938, 0.5625,
                0.5625, 0.554688, 0.570312, 0.5625, 0.570312, 0.570312, 0.578125, 0.570312, 0.585938, 0.570312,
                0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625, 0.570312, 0.578125, 0.5625, 0.554688,
                0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312, 0.570312, 0.5625, 0.585938, 0.5625,
                0.578125, 0.578125, 0.578125, 0.585938, 0.578125, 0.570312, 0.554688, 0.578125, 0.570312, 0.585938},
            std::vector<T>{
                1.20312, 1.27344, 1.375, 1.32031, 1.35156, 1.20312, 1.22656, 1.25781, 1.375, 1.23438,
                1.25, 1.21875, 1.26562, 1.23438, 1.26562, 1.26562, 1.32031, 1.26562, 1.35156, 1.28906,
                1.34375, 1.27344, 1.25781, 1.32031, 1.28906, 1.24219, 1.28125, 1.34375, 1.24219, 1.21875,
                1.28906, 1.32031, 1.35156, 1.27344, 1.28125, 1.29688, 1.28125, 1.22656, 1.35156, 1.23438,
                1.32812, 1.32812, 1.32031, 1.35938, 1.32812, 1.25781, 1.21875, 1.32031, 1.28906, 1.375}),
        TensorIteratorParams(
            5, 10, 10, 10,
            0.7f, op::RecurrentSequenceDirection::REVERSE, ngraph::helpers::TensorIteratorBody::LSTM,
            ET,
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 8.875,
                8.75, 9, 8.4375, 1.76562, 8.4375, 1.34375, 3.45312, 2.53125, 1.53125, 8.875,
                7.03125, 1.88281, 6.3125, 4.78125, 7.03125, 9.625, 4.6875, 5.8125, 2.78125, 7.21875,
                3.59375, 3.84375, 2.28125, 7.1875, 8, 8.5, 4.6875, 1.16406, 1.30469, 7.75,
                6.625, 9.875, 6.9375, 7.71875, 3.6875, 3.53125, 5, 8.125, 3, 1.92188,
                1.65625, 5, 5.21875, 9.125, 1.85938, 3.64062, 9.125, 3.59375, 2.0625, 2.15625,
                5.71875, 1.17188, 1.75, 7.125, 9.25, 2.90625, 9.1875, 3.375, 3.6875, 5.4375,
                6.25, 1.47656, 6.0625, 6.15625, 6.5, 2.3125, 9.625, 6.3125, 3.34375, 7.3125,
                3.07812, 1.92188, 5.8125, 4.71875, 9.5, 7.25, 5.4375, 4.71875, 5.875, 1.45312,
                7.875, 5.8125, 1.40625, 6.96875, 2.25, 5.625, 8.125, 9.5, 1.26562, 6.25,
                8.9375, 9.125, 5.875, 2.23438, 5.03125, 2.25, 9, 8.25, 4.375, 4.5625,
                5.84375, 2.48438, 6.875, 9.375, 4.25, 4.125, 6.125, 7.75, 6.75, 7.53125,
                2.125, 8.9375, 7.1875, 6.625, 6.8125, 7.75, 4.1875, 4.125, 7.875, 3.42188,
                4.1875, 9.0625, 7.75, 4.84375, 8.875, 9.625, 1.10156, 6.96875, 5.46875, 6.59375,
                1.66406, 2.03125, 8.0625, 9.5, 1.57812, 5.0625, 4.1875, 6.1875, 9.5, 4.6875,
                4.40625, 3.125, 7.875, 9.125, 7.9375, 6.15625, 3.71875, 1.02344, 7.9375, 6.5625,
                2.375, 3.9375, 6.1875, 5.75, 1.07812, 9, 7.375, 4.1875, 5.25, 9.125,
                7.875, 6.625, 5.1875, 1.14062, 3.40625, 9.375, 8.5, 7.1875, 5.9375, 10,
                1.625, 2.54688, 5.25, 2.21875, 7.6875, 9.375, 2.71875, 7.25, 5.1875, 1.59375,
                3.0625, 7.8125, 5.5625, 7.78125, 2.875, 9.25, 1.4375, 7.375, 5.65625, 2.125,
                2.54688, 1.17188, 4.5625, 1.23438, 1.96875, 1.25, 5.5625, 3.21875, 1.92188, 8.75,
                3.59375, 5.84375, 3.07812, 5.96875, 9.6875, 8.5625, 3.5, 2.125, 3.09375, 3.5,
                1.82031, 6.25, 6.125, 9.75, 4.75, 6.0625, 4.3125, 1.16406, 8.3125, 8.1875,
                3.59375, 3.09375, 7.4375, 8.25, 6.5, 4.5, 4.8125, 8.75, 7.75, 7.71875,
                4.84375, 6, 4.84375, 2.21875, 4.25, 1.53906, 2.375, 2.09375, 9.375, 1.39844,
                9.25, 1.96875, 8, 3.03125, 6.5625, 7.40625, 1.32031, 6.03125, 6.875, 1.10938,
                2.15625, 1.64062, 3.65625, 9.6875, 4.25, 6.125, 3.46875, 2.82812, 1.66406, 3.26562,
                2.375, 7.6875, 2.45312, 2.75, 9.4375, 6.21875, 4.3125, 9.75, 1.45312, 8.625,
                7.65625, 3.15625, 3.6875, 5.4375, 2.84375, 6.5625, 9.8125, 8.4375, 9, 2.40625,
                7.8125, 1.16406, 6.875, 1.625, 1.35938, 5.375, 8.3125, 6.4375, 7.875, 6.125,
                5.09375, 3.84375, 5.78125, 9.875, 1.98438, 6.1875, 2.3125, 4.40625, 5.5625, 5.9375,
                2.9375, 7.6875, 9.25, 7, 5.15625, 3.375, 2.1875, 1.59375, 7.875, 4.3125,
                2.90625, 6.65625, 1.67188, 2.89062, 1.85938, 7.75, 2.45312, 1.59375, 4.1875, 3.34375,
                1.85938, 8.25, 2.28125, 2.73438, 9.375, 6.75, 6.1875, 5.71875, 8.5, 9.3125,
                6.625, 3.375, 3.90625, 1.59375, 7.5625, 7.625, 5.6875, 7.9375, 7.625, 9.125,
                2.48438, 9.375, 7.1875, 1.125, 4.8125, 3.09375, 7.5625, 6.5625, 7.8125, 9.5,
                4.5625, 9.5, 9.3125, 6, 2.82812, 9.25, 1.07031, 6.75, 9.3125, 4.5,
                3.65625, 5.375, 2.5, 6.4375, 1.21875, 5.9375, 5.0625, 9.3125, 8.25, 9.25,
                4.3125, 4.5625, 6.46875, 9.625, 1.3125, 2.5625, 4.1875, 2.125, 1.70312, 2.21875,
                7.25, 5.5625, 1.10938, 1.1875, 5.125, 9.5, 9.625, 8.4375, 4, 1.13281,
                5.25, 2.57812, 1.94531, 3.98438, 5.5, 2.17188, 9, 8.25, 5.8125, 4.09375,
                3.53125, 9.4375, 4.1875, 6.25, 9.0625, 8.875, 3.17188, 8.625, 1.21875, 9.125,
                9.6875, 5.125, 4.875, 5.90625, 4.125, 8.125, 6.1875, 3.5625, 2.125, 5.40625,
                9.5, 6.375, 3.8125, 1.14062, 9.5625, 6.3125, 2.96875, 4.875, 3.23438, 8.25,
                8.75, 3.84375, 3.125, 9, 8.3125, 6.1875, 5.875, 2.65625, 2.71875, 8.0625,
                6.3125, 6.5, 1.42969, 1.48438, 1.14062, 4.78125, 1.44531, 7.125, 4.59375, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 8.875,
                8.75, 9, 8.4375, 1.76562, 8.4375, 1.34375, 3.45312, 2.53125, 1.53125, 8.875,
                7.03125, 1.88281, 6.3125, 4.78125, 7.03125, 9.625, 4.6875, 5.8125, 2.78125, 7.21875,
                3.59375, 3.84375, 2.28125, 7.1875, 8, 8.5, 4.6875, 1.16406, 1.30469, 7.75,
                6.625, 9.875, 6.9375, 7.71875, 3.6875, 3.53125, 5, 8.125, 3, 1.92188,
                1.65625, 5, 5.21875, 9.125, 1.85938, 3.64062, 9.125, 3.59375, 2.0625, 2.15625,
                5.71875, 1.17188, 1.75, 7.125, 9.25, 2.90625, 9.1875, 3.375, 3.6875, 5.4375,
                6.25, 1.47656, 6.0625, 6.15625, 6.5, 2.3125, 9.625, 6.3125, 3.34375, 7.3125,
                3.07812, 1.92188, 5.8125, 4.71875, 9.5, 7.25, 5.4375, 4.71875, 5.875, 1.45312,
                7.875, 5.8125, 1.40625, 6.96875, 2.25, 5.625, 8.125, 9.5, 1.26562, 6.25,
                8.9375, 9.125, 5.875, 2.23438, 5.03125, 2.25, 9, 8.25, 4.375, 4.5625,
                5.84375, 2.48438, 6.875, 9.375, 4.25, 4.125, 6.125, 7.75, 6.75, 7.53125,
                2.125, 8.9375, 7.1875, 6.625, 6.8125, 7.75, 4.1875, 4.125, 7.875, 3.42188,
                4.1875, 9.0625, 7.75, 4.84375, 8.875, 9.625, 1.10156, 6.96875, 5.46875, 6.59375,
                1.66406, 2.03125, 8.0625, 9.5, 1.57812, 5.0625, 4.1875, 6.1875, 9.5, 4.6875,
                4.40625, 3.125, 7.875, 9.125, 7.9375, 6.15625, 3.71875, 1.02344, 7.9375, 6.5625,
                2.375, 3.9375, 6.1875, 5.75, 1.07812, 9, 7.375, 4.1875, 5.25, 9.125,
                7.875, 6.625, 5.1875, 1.14062, 3.40625, 9.375, 8.5, 7.1875, 5.9375, 10,
                1.625, 2.54688, 5.25, 2.21875, 7.6875, 9.375, 2.71875, 7.25, 5.1875, 1.59375,
                3.0625, 7.8125, 5.5625, 7.78125, 2.875, 9.25, 1.4375, 7.375, 5.65625, 2.125,
                2.54688, 1.17188, 4.5625, 1.23438, 1.96875, 1.25, 5.5625, 3.21875, 1.92188, 8.75,
                3.59375, 5.84375, 3.07812, 5.96875, 9.6875, 8.5625, 3.5, 2.125, 3.09375, 3.5,
                1.82031, 6.25, 6.125, 9.75, 4.75, 6.0625, 4.3125, 1.16406, 8.3125, 8.1875,
                3.59375, 3.09375, 7.4375, 8.25, 6.5, 4.5, 4.8125, 8.75, 7.75, 7.71875,
                4.84375, 6, 4.84375, 2.21875, 4.25, 1.53906, 2.375, 2.09375, 9.375, 1.39844,
                9.25, 1.96875, 8, 3.03125, 6.5625, 7.40625, 1.32031, 6.03125, 6.875, 1.10938,
                2.15625, 1.64062, 3.65625, 9.6875, 4.25, 6.125, 3.46875, 2.82812, 1.66406, 3.26562,
                2.375, 7.6875, 2.45312, 2.75, 9.4375, 6.21875, 4.3125, 9.75, 1.45312, 8.625,
                7.65625, 3.15625, 3.6875, 5.4375, 2.84375, 6.5625, 9.8125, 8.4375, 9, 2.40625,
                7.8125, 1.16406, 6.875, 1.625, 1.35938, 5.375, 8.3125, 6.4375, 7.875, 6.125,
                5.09375, 3.84375, 5.78125, 9.875, 1.98438, 6.1875, 2.3125, 4.40625, 5.5625, 5.9375,
                2.9375, 7.6875, 9.25, 7, 5.15625, 3.375, 2.1875, 1.59375, 7.875, 4.3125,
                2.90625, 6.65625, 1.67188, 2.89062, 1.85938, 7.75, 2.45312, 1.59375, 4.1875, 3.34375,
                1.85938, 8.25, 2.28125, 2.73438, 9.375, 6.75, 6.1875, 5.71875, 8.5, 9.3125,
                6.625, 3.375, 3.90625, 1.59375, 7.5625, 7.625, 5.6875, 7.9375, 7.625, 9.125,
                2.48438, 9.375, 7.1875, 1.125, 4.8125, 3.09375, 7.5625, 6.5625, 7.8125, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 2.78125,
                8, 8.1875, 7.4375, 9.6875, 8.25, 3.8125, 1.82812, 7.21875, 5.65625, 8.875,
                8.75, 9, 8.4375, 1.76562, 8.4375, 1.34375, 3.45312, 2.53125, 1.53125, 8.875,
                7.03125, 1.88281, 6.3125, 4.78125, 7.03125, 9.625, 4.6875, 5.8125, 2.78125, 7.21875,
                3.59375, 3.84375, 2.28125, 7.1875, 8, 8.5, 4.6875, 1.16406, 1.30469, 7.75,
                6.625, 9.875, 6.9375, 7.71875, 3.6875, 3.53125, 5, 8.125, 3, 1.92188,
                1.65625, 5, 5.21875, 9.125, 1.85938, 3.64062, 9.125, 3.59375, 2.0625, 2.15625,
                5.71875, 1.17188, 1.75, 7.125, 9.25, 2.90625, 9.1875, 3.375, 3.6875, 5.4375,
                6.25, 1.47656, 6.0625, 6.15625, 6.5, 2.3125, 9.625, 6.3125, 3.34375, 7.3125,
                3.07812, 1.92188, 5.8125, 4.71875, 9.5, 7.25, 5.4375, 4.71875, 5.875, 1.45312,
                7.875, 5.8125, 1.40625, 6.96875, 2.25, 5.625, 8.125, 9.5, 1.26562, 6.25,
                8.9375, 9.125, 5.875, 2.23438, 5.03125, 2.25, 9, 8.25, 4.375, 4.5625,
                5.84375, 2.48438, 6.875, 9.375, 4.25, 4.125, 6.125, 7.75, 6.75, 7.53125,
                2.125, 8.9375, 7.1875, 6.625, 6.8125, 7.75, 4.1875, 4.125, 7.875, 3.42188,
                4.1875, 9.0625, 7.75, 4.84375, 8.875, 9.625, 1.10156, 6.96875, 5.46875, 6.59375,
                1.66406, 2.03125, 8.0625, 9.5, 1.57812, 5.0625, 4.1875, 6.1875, 9.5, 4.6875,
                4.40625, 3.125, 7.875, 9.125, 7.9375, 6.15625, 3.71875, 1.02344, 7.9375, 6.5625,
                2.375, 3.9375, 6.1875, 5.75, 1.07812, 9, 7.375, 4.1875, 5.25, 9.125,
                7.875, 6.625, 5.1875, 1.14062, 3.40625, 9.375, 8.5, 7.1875, 5.9375, 10,
                1.625, 2.54688, 5.25, 2.21875, 7.6875, 9.375, 2.71875, 7.25, 5.1875, 1.59375,
                3.0625, 7.8125, 5.5625, 7.78125, 2.875, 9.25, 1.4375, 7.375, 5.65625, 2.125,
                2.54688, 1.17188, 4.5625, 1.23438, 1.96875, 1.25, 5.5625, 3.21875, 1.92188, 8.75,
                3.59375, 5.84375, 3.07812, 5.96875, 9.6875, 8.5625, 3.5, 2.125, 3.09375, 3.5,
                1.82031, 6.25, 6.125, 9.75, 4.75, 6.0625, 4.3125, 1.16406, 8.3125, 8.1875,
                3.59375, 3.09375, 7.4375, 8.25, 6.5, 4.5, 4.8125, 8.75, 7.75, 7.71875,
                4.84375, 6, 4.84375, 2.21875, 4.25, 1.53906, 2.375, 2.09375, 9.375, 1.39844,
                9.25, 1.96875, 8, 3.03125, 6.5625, 7.40625, 1.32031, 6.03125, 6.875, 1.10938,
                2.15625, 1.64062, 3.65625, 9.6875, 4.25, 6.125, 3.46875, 2.82812, 1.66406, 3.26562,
                2.375, 7.6875, 2.45312, 2.75, 9.4375, 6.21875, 4.3125, 9.75, 1.45312, 8.625,
                7.65625, 3.15625, 3.6875, 5.4375, 2.84375, 6.5625, 9.8125, 8.4375, 9, 2.40625,
                7.8125, 1.16406, 6.875, 1.625, 1.35938, 5.375, 8.3125, 6.4375, 7.875, 6.125,
                5.09375, 3.84375, 5.78125, 9.875, 1.98438, 6.1875, 2.3125, 4.40625, 5.5625, 5.9375,
                2.9375, 7.6875, 9.25, 7, 5.15625, 3.375, 2.1875, 1.59375, 7.875, 4.3125,
                2.90625, 6.65625, 1.67188, 2.89062, 1.85938, 7.75, 2.45312, 1.59375, 4.1875, 3.34375,
                1.85938, 8.25, 2.28125, 2.73438, 9.375, 6.75, 6.1875, 5.71875, 8.5, 9.3125,
                6.625, 3.375, 3.90625, 1.59375, 7.5625, 7.625, 5.6875, 7.9375, 7.625, 9.125,
                2.48438, 9.375, 7.1875, 1.125, 4.8125, 3.09375, 7.5625, 6.5625, 7.8125, 10},
            std::vector<T>{
                1, 4.75, 10, 7.46875, 9.375, 1, 2.15625, 3.71875, 10, 2.3125,
                3.125, 1.82812, 4.5625, 2.67188, 4.5, 4.125, 7, 4.5625, 9.375, 5.84375,
                8.625, 4.75, 3.8125, 7.15625, 5.71875, 2.84375, 5, 8.875, 3.0625, 1.25,
                5.8125, 7.03125, 9.25, 4.75, 5.125, 6, 4.875, 2.25, 9.4375, 10},
            std::vector<T>{
                0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625, 0.570312, 0.585938, 0.5625,
                0.554688, 0.570312, 0.59375, 0.585938, 0.59375, 0.554688, 0.5625, 0.570312, 0.59375, 0.5625,
                0.554688, 0.585938, 0.609375, 0.59375, 0.609375, 0.554688, 0.5625, 0.578125, 0.609375, 0.570312,
                0.554688, 0.59375, 0.625, 0.609375, 0.625, 0.554688, 0.570312, 0.585938, 0.625, 0.570312,
                0.554688, 0.601562, 0.640625, 0.625, 0.640625, 0.554688, 0.570312, 0.59375, 0.640625, 0.578125,
                0.554688, 0.617188, 0.65625, 0.640625, 0.648438, 0.554688, 0.578125, 0.609375, 0.65625, 0.585938,
                0.546875, 0.632812, 0.664062, 0.65625, 0.664062, 0.546875, 0.585938, 0.625, 0.664062, 0.59375,
                0.546875, 0.648438, 0.667969, 0.664062, 0.667969, 0.546875, 0.601562, 0.640625, 0.667969, 0.609375,
                0.539062, 0.664062, 0.667969, 0.667969, 0.667969, 0.539062, 0.617188, 0.65625, 0.667969, 0.625,
                0.523438, 0.667969, 0.667969, 0.667969, 0.667969, 0.523438, 0.632812, 0.664062, 0.667969, 0.640625,
                0.5625, 0.554688, 0.570312, 0.5625, 0.570312, 0.570312, 0.578125, 0.570312, 0.585938, 0.570312,
                0.570312, 0.5625, 0.570312, 0.5625, 0.570312, 0.570312, 0.585938, 0.570312, 0.59375, 0.578125,
                0.570312, 0.5625, 0.578125, 0.570312, 0.578125, 0.578125, 0.59375, 0.578125, 0.609375, 0.585938,
                0.578125, 0.5625, 0.585938, 0.570312, 0.585938, 0.585938, 0.609375, 0.585938, 0.625, 0.601562,
                0.585938, 0.570312, 0.601562, 0.578125, 0.601562, 0.601562, 0.625, 0.601562, 0.640625, 0.617188,
                0.601562, 0.570312, 0.617188, 0.585938, 0.617188, 0.617188, 0.640625, 0.617188, 0.648438, 0.632812,
                0.617188, 0.578125, 0.632812, 0.601562, 0.632812, 0.632812, 0.65625, 0.632812, 0.664062, 0.648438,
                0.632812, 0.585938, 0.648438, 0.617188, 0.648438, 0.648438, 0.664062, 0.648438, 0.667969, 0.65625,
                0.648438, 0.601562, 0.664062, 0.632812, 0.664062, 0.65625, 0.667969, 0.664062, 0.667969, 0.664062,
                0.65625, 0.617188, 0.664062, 0.648438, 0.664062, 0.664062, 0.667969, 0.664062, 0.667969, 0.667969,
                0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625, 0.570312, 0.578125, 0.5625, 0.554688,
                0.585938, 0.570312, 0.570312, 0.585938, 0.578125, 0.570312, 0.578125, 0.585938, 0.570312, 0.554688,
                0.601562, 0.585938, 0.578125, 0.59375, 0.585938, 0.570312, 0.585938, 0.601562, 0.570312, 0.554688,
                0.617188, 0.59375, 0.585938, 0.609375, 0.601562, 0.578125, 0.59375, 0.617188, 0.578125, 0.554688,
                0.632812, 0.601562, 0.59375, 0.625, 0.617188, 0.585938, 0.609375, 0.632812, 0.585938, 0.554688,
                0.648438, 0.617188, 0.609375, 0.640625, 0.632812, 0.59375, 0.625, 0.648438, 0.59375, 0.554688,
                0.65625, 0.632812, 0.625, 0.65625, 0.648438, 0.609375, 0.640625, 0.664062, 0.609375, 0.554688,
                0.664062, 0.648438, 0.640625, 0.664062, 0.65625, 0.625, 0.65625, 0.664062, 0.625, 0.554688,
                0.667969, 0.664062, 0.65625, 0.667969, 0.664062, 0.640625, 0.664062, 0.667969, 0.640625, 0.5625,
                0.667969, 0.667969, 0.664062, 0.667969, 0.667969, 0.648438, 0.667969, 0.667969, 0.65625, 0.5625,
                0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312, 0.570312, 0.5625, 0.585938, 0.5625,
                0.578125, 0.585938, 0.59375, 0.570312, 0.578125, 0.578125, 0.578125, 0.5625, 0.59375, 0.5625,
                0.585938, 0.59375, 0.601562, 0.585938, 0.585938, 0.585938, 0.585938, 0.5625, 0.609375, 0.570312,
                0.601562, 0.609375, 0.617188, 0.59375, 0.59375, 0.601562, 0.59375, 0.570312, 0.625, 0.570312,
                0.617188, 0.625, 0.632812, 0.601562, 0.609375, 0.617188, 0.609375, 0.570312, 0.640625, 0.585938,
                0.632812, 0.640625, 0.648438, 0.617188, 0.625, 0.632812, 0.625, 0.585938, 0.648438, 0.59375,
                0.648438, 0.65625, 0.664062, 0.632812, 0.640625, 0.648438, 0.640625, 0.59375, 0.664062, 0.601562,
                0.65625, 0.664062, 0.667969, 0.648438, 0.65625, 0.65625, 0.648438, 0.601562, 0.667969, 0.617188,
                0.664062, 0.667969, 0.667969, 0.664062, 0.664062, 0.664062, 0.664062, 0.617188, 0.667969, 0.632812,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.632812, 0.667969, 0.648438,
                0.578125, 0.578125, 0.578125, 0.585938, 0.578125, 0.570312, 0.554688, 0.578125, 0.570312, 0.585938,
                0.585938, 0.585938, 0.585938, 0.59375, 0.585938, 0.570312, 0.5625, 0.585938, 0.578125, 0.59375,
                0.601562, 0.601562, 0.59375, 0.609375, 0.601562, 0.578125, 0.5625, 0.59375, 0.585938, 0.609375,
                0.617188, 0.617188, 0.609375, 0.625, 0.617188, 0.585938, 0.5625, 0.609375, 0.601562, 0.625,
                0.632812, 0.632812, 0.625, 0.640625, 0.632812, 0.59375, 0.570312, 0.625, 0.617188, 0.640625,
                0.648438, 0.648438, 0.640625, 0.65625, 0.648438, 0.609375, 0.570312, 0.640625, 0.632812, 0.65625,
                0.65625, 0.65625, 0.65625, 0.664062, 0.65625, 0.625, 0.578125, 0.65625, 0.648438, 0.664062,
                0.664062, 0.664062, 0.664062, 0.667969, 0.664062, 0.640625, 0.585938, 0.664062, 0.65625, 0.667969,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.65625, 0.601562, 0.667969, 0.664062, 0.667969,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.664062, 0.617188, 0.667969, 0.667969, 0.667969},
            std::vector<T>{
                0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625, 0.570312, 0.585938, 0.5625,
                0.5625, 0.554688, 0.570312, 0.5625, 0.570312, 0.570312, 0.578125, 0.570312, 0.585938, 0.570312,
                0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625, 0.570312, 0.578125, 0.5625, 0.554688,
                0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312, 0.570312, 0.5625, 0.585938, 0.5625,
                0.578125, 0.578125, 0.578125, 0.585938, 0.578125, 0.570312, 0.554688, 0.578125, 0.570312, 0.585938},
            std::vector<T>{
                1.20312, 1.27344, 1.375, 1.32031, 1.35156, 1.20312, 1.22656, 1.25781, 1.375, 1.23438,
                1.25, 1.21875, 1.26562, 1.23438, 1.26562, 1.26562, 1.32031, 1.26562, 1.35156, 1.28906,
                1.34375, 1.27344, 1.25781, 1.32031, 1.28906, 1.24219, 1.28125, 1.34375, 1.24219, 1.21875,
                1.28906, 1.32031, 1.35156, 1.27344, 1.28125, 1.29688, 1.28125, 1.22656, 1.35156, 1.23438,
                1.32812, 1.32812, 1.32031, 1.35938, 1.32812, 1.25781, 1.21875, 1.32031, 1.28906, 1.375}),
    };
    return params;
}

std::vector<TensorIteratorParams> generateCombinedParams() {
    const std::vector<std::vector<TensorIteratorParams>> generatedParams {
        generateParams<element::Type_t::f64>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f16>(),
        generateParamsBF16<element::Type_t::bf16>(),
    };
    std::vector<TensorIteratorParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_TensorIterator_With_Hardcoded_Refs, ReferenceTensorIteratorTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceTensorIteratorTest::getTestCaseName);
} // namespace
