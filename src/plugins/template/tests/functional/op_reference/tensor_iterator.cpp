// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tensor_iterator.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {
struct TIFunctionalBase {
    virtual std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& ti_inputs,
                                                       const std::vector<reference_tests::Tensor>& results) = 0;

    TIFunctionalBase() = default;

    virtual ~TIFunctionalBase() = default;
};

struct TIDynamicInputs : public TIFunctionalBase {
    std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& ti_inputs,
                                               const std::vector<reference_tests::Tensor>& results) override {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto M = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto Xi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto Yi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto M_body = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto body_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);

        auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 3);
        auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        // Body
        auto sum = std::make_shared<ov::op::v1::Add>(Xi, Yi);
        auto Zo = std::make_shared<ov::op::v1::Multiply>(sum, M_body);
        auto body =
            std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo}, ov::ParameterVector{Xi, Yi, M_body});

        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
        tensor_iterator->set_function(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(M_body, M, Zo);

        // Output 0 is last Zo
        auto out1 = tensor_iterator->get_iter_value(Zo, -1);
        return std::make_shared<ov::Model>(ov::OutputVector{out1}, ov::ParameterVector{X, Y, M});
    }
};

struct TensorIteratorParams {
    TensorIteratorParams(const std::shared_ptr<TIFunctionalBase>& functional,
                         const std::vector<reference_tests::Tensor>& ti_inputs,
                         const std::vector<reference_tests::Tensor>& expected_results,
                         const std::string& test_case_name)
        : function(functional),
          inputs(ti_inputs),
          expected_results(expected_results),
          test_case_name(test_case_name) {}

    std::shared_ptr<TIFunctionalBase> function;
    std::vector<reference_tests::Tensor> inputs;
    std::vector<reference_tests::Tensor> expected_results;
    std::string test_case_name;
};

class ReferenceTILayerTest : public testing::TestWithParam<TensorIteratorParams>,
                             public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        legacy_compare = true;
        auto params = GetParam();
        function = params.function->create_function(params.inputs, params.expected_results);
        inputData.reserve(params.inputs.size());
        refOutData.reserve(params.expected_results.size());
        for (auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        for (auto& expected_tensor : params.expected_results) {
            refOutData.push_back(expected_tensor.data);
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }
};

TEST_P(ReferenceTILayerTest, TensorIteratorWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_TensorIterator_With_Hardcoded_Refs,
                         ReferenceTILayerTest,
                         ::testing::Values(TensorIteratorParams(
                             std::make_shared<TIDynamicInputs>(),
                             std::vector<reference_tests::Tensor>{
                                 reference_tests::Tensor(ov::element::f32, ov::Shape{1, 2}, std::vector<float>{2, 3}),
                                 reference_tests::Tensor(ov::element::f32, ov::Shape{2, 1}, std::vector<float>{4, 5}),
                                 reference_tests::Tensor(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{5})},
                             std::vector<reference_tests::Tensor>{
                                 reference_tests::Tensor(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{240})},
                             "tensor_iterator_dynamic_inputs")),
                         ReferenceTILayerTest::getTestCaseName);

enum TensorIteratorBodyType {
    RNN,
    GRU,
    LSTM,
};

struct TensorIteratorStaticParams;

struct TIStaticFunctionalBase {
    virtual std::shared_ptr<ov::Model> create_function(const TensorIteratorStaticParams& params) = 0;
    TIStaticFunctionalBase() = default;
    virtual ~TIStaticFunctionalBase() = default;
};

struct TensorIteratorStaticParams {
    template <class T>
    TensorIteratorStaticParams(const std::shared_ptr<TIStaticFunctionalBase>& functional,
                               const size_t batchSize,
                               const size_t inputSize,
                               const size_t hiddenSize,
                               const size_t seqLength,
                               const float clip,
                               const ov::op::RecurrentSequenceDirection& direction,
                               const TensorIteratorBodyType& body_type,
                               const ov::element::Type_t& iType,
                               const std::vector<T>& XValues,
                               const std::vector<T>& H_tValues,
                               const std::vector<T>& C_tValues,
                               const std::vector<int64_t>& S_tValues,
                               const std::vector<T>& WValues,
                               const std::vector<T>& RValues,
                               const std::vector<T>& BValues,
                               const std::vector<T>& YValues,
                               const std::vector<T>& HoValues,
                               const std::vector<T>& CoValues,
                               const std::string& testcaseName = "")
        : function(functional),
          batchSize(batchSize),
          inputSize(inputSize),
          hiddenSize(hiddenSize),
          seqLength(seqLength),
          clip(clip),
          body_type(body_type),
          direction(direction),
          iType(iType),
          oType(iType),
          testcaseName(testcaseName) {
        switch (body_type) {
        case TensorIteratorBodyType::LSTM: {
            ov::Shape XShape = ov::Shape{batchSize, seqLength, inputSize};
            ov::Shape H_tShape = ov::Shape{batchSize, hiddenSize};
            ov::Shape C_tShape = ov::Shape{batchSize, hiddenSize};
            ov::Shape S_tShape = ov::Shape{batchSize};
            ov::Shape WShape = ov::Shape{4 * hiddenSize, inputSize};
            ov::Shape RShape = ov::Shape{4 * hiddenSize, hiddenSize};
            ov::Shape BShape = ov::Shape{4 * hiddenSize};
            ov::Shape YShape = ov::Shape{batchSize, seqLength, hiddenSize};
            ov::Shape HoShape = ov::Shape{batchSize, hiddenSize};
            ov::Shape CoShape = ov::Shape{batchSize, hiddenSize};

            X = reference_tests::Tensor(XShape, iType, XValues);
            H_t = reference_tests::Tensor(H_tShape, iType, H_tValues);
            C_t = reference_tests::Tensor(C_tShape, iType, C_tValues);
            S_t = reference_tests::Tensor(S_tShape, ov::element::Type_t::i64, S_tValues);
            W = reference_tests::Tensor(WShape, iType, WValues);
            R = reference_tests::Tensor(RShape, iType, RValues);
            B = reference_tests::Tensor(BShape, iType, BValues);
            Y = reference_tests::Tensor(YShape, oType, YValues);
            Ho = reference_tests::Tensor(HoShape, oType, HoValues);
            Co = reference_tests::Tensor(CoShape, oType, CoValues);
            break;
        }
        case TensorIteratorBodyType::GRU: {
            ov::Shape XShape = ov::Shape{batchSize, seqLength, inputSize};
            ov::Shape H_tShape = ov::Shape{batchSize, hiddenSize};
            ov::Shape S_tShape = ov::Shape{batchSize};
            ov::Shape WShape = ov::Shape{3 * hiddenSize, inputSize};
            ov::Shape RShape = ov::Shape{3 * hiddenSize, hiddenSize};
            ov::Shape BShape = ov::Shape{3 * hiddenSize};
            ov::Shape YShape = ov::Shape{batchSize, seqLength, hiddenSize};
            ov::Shape HoShape = ov::Shape{batchSize, hiddenSize};

            X = reference_tests::Tensor(XShape, iType, XValues);
            H_t = reference_tests::Tensor(H_tShape, iType, H_tValues);
            S_t = reference_tests::Tensor(S_tShape, ov::element::Type_t::i64, S_tValues);
            W = reference_tests::Tensor(WShape, iType, WValues);
            R = reference_tests::Tensor(RShape, iType, RValues);
            B = reference_tests::Tensor(BShape, iType, BValues);
            Y = reference_tests::Tensor(YShape, oType, YValues);
            Ho = reference_tests::Tensor(HoShape, oType, HoValues);
            break;
        }
        case TensorIteratorBodyType::RNN: {
            ov::Shape XShape = ov::Shape{batchSize, seqLength, inputSize};
            ov::Shape H_tShape = ov::Shape{batchSize, hiddenSize};
            ov::Shape S_tShape = ov::Shape{batchSize};
            ov::Shape WShape = ov::Shape{hiddenSize, inputSize};
            ov::Shape RShape = ov::Shape{hiddenSize, hiddenSize};
            ov::Shape BShape = ov::Shape{hiddenSize};
            ov::Shape YShape = ov::Shape{batchSize, seqLength, hiddenSize};
            ov::Shape HoShape = ov::Shape{batchSize, hiddenSize};

            X = reference_tests::Tensor(XShape, iType, XValues);
            H_t = reference_tests::Tensor(H_tShape, iType, H_tValues);
            S_t = reference_tests::Tensor(S_tShape, ov::element::Type_t::i64, S_tValues);
            W = reference_tests::Tensor(WShape, iType, WValues);
            R = reference_tests::Tensor(RShape, iType, RValues);
            B = reference_tests::Tensor(BShape, iType, BValues);
            Y = reference_tests::Tensor(YShape, oType, YValues);
            Ho = reference_tests::Tensor(HoShape, oType, HoValues);
            break;
        }
        }
    }

    std::shared_ptr<TIStaticFunctionalBase> function;

    size_t batchSize;
    size_t inputSize;
    size_t hiddenSize;
    size_t seqLength;
    size_t sequenceAxis = 1;
    float clip;
    TensorIteratorBodyType body_type;
    ov::op::RecurrentSequenceDirection direction;

    ov::element::Type_t iType;
    ov::element::Type_t oType;

    reference_tests::Tensor X;
    reference_tests::Tensor H_t;
    reference_tests::Tensor C_t;
    reference_tests::Tensor S_t;
    reference_tests::Tensor W;
    reference_tests::Tensor R;
    reference_tests::Tensor B;
    reference_tests::Tensor Y;
    reference_tests::Tensor Ho;
    reference_tests::Tensor Co;
    std::string testcaseName;
};

struct TIStaticInputs : public TIStaticFunctionalBase {
    std::shared_ptr<ov::Model> create_function(const TensorIteratorStaticParams& params) override {
        std::vector<ov::Shape> inputShapes;
        std::shared_ptr<ov::Model> function;
        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();

        // Each case consist of 3 steps:
        // 1. Create TensorIterator body.
        // 2. Set PortMap
        // 3. Create outer function
        auto axis =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{1},
                                                   std::vector<int64_t>{static_cast<int64_t>(params.sequenceAxis)});
        switch (params.body_type) {
        case TensorIteratorBodyType::LSTM: {
            inputShapes = {
                {{params.batchSize, params.seqLength, params.inputSize},  // X
                 {params.batchSize, params.hiddenSize},                   // H_i
                 {params.batchSize, params.hiddenSize},                   // C_i
                 {4 * params.hiddenSize, params.inputSize},               // W
                 {4 * params.hiddenSize, params.inputSize},               // R
                 {4 * params.inputSize}},                                 // B
            };
            if (params.sequenceAxis == 0) {
                // swap params.batchSize and params.seqLength
                std::swap(inputShapes[0][0], inputShapes[0][1]);
            }
            ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[0]),
                                             std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[1]),
                                             std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[2])};

            // 1. Create TensorIterator body.
            inputShapes[0][params.sequenceAxis] = 1;  // sliced dimension
            ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[0]),
                                            std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[1]),
                                            std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[2])};
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
            ov::OutputVector out_vector = {squeeze, body_params[1], body_params[2]};

            auto W = std::make_shared<ov::op::v0::Constant>(params.W.type, params.W.shape, params.W.data.data());
            auto R = std::make_shared<ov::op::v0::Constant>(params.R.type, params.R.shape, params.R.data.data());
            auto B = std::make_shared<ov::op::v0::Constant>(params.B.type, params.B.shape, params.B.data.data());
            auto lstm_cell = std::make_shared<ov::op::v4::LSTMCell>(out_vector[0],
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

            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(lstm_cell->output(0), axis);
            ov::ResultVector results{std::make_shared<ov::op::v0::Result>(unsqueeze),
                                     std::make_shared<ov::op::v0::Result>(lstm_cell->output(0)),
                                     std::make_shared<ov::op::v0::Result>(lstm_cell->output(1))};
            auto body = std::make_shared<ov::Model>(results, body_params, "lstm_cell");
            tensor_iterator->set_function(body);

            // 2. Set PortMap
            if (params.direction == ov::op::RecurrentSequenceDirection::FORWARD) {
                tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, params.sequenceAxis);
                tensor_iterator->get_concatenated_slices(results[0], 0, 1, 1, -1, params.sequenceAxis);
            } else if (params.direction == ov::op::RecurrentSequenceDirection::REVERSE) {
                tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, params.sequenceAxis);
                tensor_iterator->get_concatenated_slices(results[0], -1, -1, 1, 0, params.sequenceAxis);
            } else {
                OPENVINO_ASSERT(false, "Bidirectional case is not supported.");
            }

            tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[1]);
            tensor_iterator->set_merged_input(body_params[2], outer_params[2], results[2]);
            tensor_iterator->get_iter_value(results[1]);
            tensor_iterator->get_iter_value(results[2]);

            // 3. Outer function
            function = std::make_shared<ov::Model>(
                ov::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1), tensor_iterator->output(2)},
                outer_params);
            break;
        }
        case TensorIteratorBodyType::GRU: {
            inputShapes = {
                {{params.batchSize, params.seqLength, params.inputSize},  // X
                 {params.batchSize, params.inputSize},                    // H_i
                 {3 * params.hiddenSize, params.inputSize},               // W
                 {3 * params.hiddenSize, params.hiddenSize},              // R
                 {3 * params.hiddenSize}},                                // B
            };
            if (params.sequenceAxis == 0) {
                // swap params.batchSize and params.seqLength
                std::swap(inputShapes[0][0], inputShapes[0][1]);
            }
            ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[0]),
                                             std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[1])};

            // 1. Create TensorIterator body.
            inputShapes[0][params.sequenceAxis] = 1;  // sliced dimension
            ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[0]),
                                            std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[1])};

            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
            ov::OutputVector out_vector = {squeeze, body_params[1]};

            auto W = std::make_shared<ov::op::v0::Constant>(params.W.type, params.W.shape, params.W.data.data());
            auto R = std::make_shared<ov::op::v0::Constant>(params.R.type, params.R.shape, params.R.data.data());
            auto B = std::make_shared<ov::op::v0::Constant>(params.B.type, params.B.shape, params.B.data.data());
            auto gru_cell = std::make_shared<ov::op::v3::GRUCell>(out_vector[0],
                                                                  out_vector[1],
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  params.hiddenSize,
                                                                  std::vector<std::string>{"sigmoid", "tanh"},
                                                                  std::vector<float>{},
                                                                  std::vector<float>{},
                                                                  params.clip,
                                                                  false);

            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gru_cell->output(0), axis);
            ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gru_cell->output(0)),
                                     std::make_shared<ov::op::v0::Result>(unsqueeze)};
            auto body = std::make_shared<ov::Model>(results, body_params, "gru_cell");
            tensor_iterator->set_function(body);

            // 2. Set PortMap
            if (params.direction == ov::op::RecurrentSequenceDirection::FORWARD) {
                tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, params.sequenceAxis);
                tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, params.sequenceAxis);
            } else if (params.direction == ov::op::RecurrentSequenceDirection::REVERSE) {
                tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, params.sequenceAxis);
                tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, params.sequenceAxis);
            } else {
                OPENVINO_ASSERT(false, "Bidirectional case is not supported.");
            }

            tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
            tensor_iterator->get_iter_value(results[0]);

            // 3. Outer function
            function =
                std::make_shared<ov::Model>(ov::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)},
                                            outer_params);
            break;
        }
        case TensorIteratorBodyType::RNN: {
            inputShapes = {
                {{params.batchSize, params.seqLength, params.inputSize},  // X
                 {params.batchSize, params.inputSize},                    // H_i
                 {params.hiddenSize, params.inputSize},                   // W
                 {params.hiddenSize, params.hiddenSize},                  // R
                 {params.hiddenSize}},                                    // B
            };
            if (params.sequenceAxis == 0) {
                // swap params.batchSize and params.seqLength
                std::swap(inputShapes[0][0], inputShapes[0][1]);
            }
            ov::ParameterVector outer_params{std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[0]),
                                             std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[1])};

            // 1. Create TensorIterator body.
            inputShapes[0][params.sequenceAxis] = 1;  // sliced dimension
            ov::ParameterVector body_params{std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[0]),
                                            std::make_shared<ov::op::v0::Parameter>(params.iType, inputShapes[1])};
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(body_params[0], axis);
            ov::OutputVector out_vector = {squeeze, body_params[1]};

            auto W = std::make_shared<ov::op::v0::Constant>(params.W.type, params.W.shape, params.W.data.data());
            auto R = std::make_shared<ov::op::v0::Constant>(params.R.type, params.R.shape, params.R.data.data());
            auto B = std::make_shared<ov::op::v0::Constant>(params.B.type, params.B.shape, params.B.data.data());
            auto rnn_cell = std::make_shared<ov::op::v0::RNNCell>(out_vector[0],
                                                                  out_vector[1],
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  params.hiddenSize,
                                                                  std::vector<std::string>{"tanh"},
                                                                  std::vector<float>{},
                                                                  std::vector<float>{},
                                                                  params.clip);

            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(rnn_cell->output(0), axis);
            ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rnn_cell),
                                     std::make_shared<ov::op::v0::Result>(unsqueeze)};
            auto body = std::make_shared<ov::Model>(results, body_params, "rnn_cell");
            tensor_iterator->set_function(body);

            // 2. Set PortMap
            if (params.direction == ov::op::RecurrentSequenceDirection::FORWARD) {
                tensor_iterator->set_sliced_input(body_params[0], outer_params[0], 0, 1, 1, -1, params.sequenceAxis);
                tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, params.sequenceAxis);
            } else if (params.direction == ov::op::RecurrentSequenceDirection::REVERSE) {
                tensor_iterator->set_sliced_input(body_params[0], outer_params[0], -1, -1, 1, 0, params.sequenceAxis);
                tensor_iterator->get_concatenated_slices(results[1], -1, -1, 1, 0, params.sequenceAxis);
            } else {
                OPENVINO_ASSERT(false, "Bidirectional case is not supported.");
            }

            tensor_iterator->set_merged_input(body_params[1], outer_params[1], results[0]);
            tensor_iterator->get_iter_value(results[0]);

            // 3. Outer function
            function =
                std::make_shared<ov::Model>(ov::OutputVector{tensor_iterator->output(0), tensor_iterator->output(1)},
                                            outer_params);
            break;
        }
        }
        return function;
    }
};

class ReferenceTILayerStaticTest : public testing::TestWithParam<TensorIteratorStaticParams>,
                                   public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        legacy_compare = true;
        auto params = GetParam();
        function = params.function->create_function(params);
        if (params.body_type == TensorIteratorBodyType::LSTM) {
            inputData = {params.X.data, params.H_t.data, params.C_t.data};
            refOutData = {params.Y.data, params.Ho.data, params.Co.data};
        } else {
            inputData = {params.X.data, params.H_t.data};
            refOutData = {params.Y.data, params.Ho.data};
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorStaticParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bSize=" << param.batchSize;
        result << "_iSize=" << param.inputSize;
        result << "_hSize=" << param.hiddenSize;
        result << "_sLength=" << param.seqLength;
        result << "_clip=" << param.clip;
        result << "_body=" << param.body_type;
        result << "_xType=" << param.X.type;
        result << "_xShape=" << param.X.shape;
        if (param.testcaseName != "") {
            result << "_direction=" << param.direction;
            result << "_" << param.testcaseName;
        } else {
            result << "_direction=" << param.direction;
        }
        return result.str();
    }
};

TEST_P(ReferenceTILayerStaticTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t ET>
std::vector<TensorIteratorStaticParams> generateParams() {
    using T = typename ov::element_type_traits<ET>::value_type;

    std::vector<TensorIteratorStaticParams> params{
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::FORWARD,
            TensorIteratorBodyType::LSTM,
            ET,
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 9.13242, 2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457,
                5.55154, 7.24966, 5.1128,  4.25147, 8.34407, 1.4123,  4.49045, 5.12671, 7.62159, 9.18673, 3.49665,
                8.35992, 6.90684, 1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283,  9.88249,
                4.48422, 8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155,  9.94928, 6.88321, 1.33685,
                7.4718,  7.19305, 6.47932, 1.9559,  3.52616, 7.98593, 9.0115,  5.59539, 7.44137, 1.70001, 6.53774,
                8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153, 5.4073,  9.00437,
                8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473, 7.95084, 2.11852, 9.28916,
                1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549, 3.75218, 8.98228, 8.98158, 5.63695,
                3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035, 8.37644, 8.54954, 5.70516, 2.45744, 9.54079,
                1.53504, 8.9785,  6.1691,  4.40962, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                           6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                           6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 10},
            std::vector<T>{
                0.528016, 0.668187, 0.668186, 0.635471, 0.668187, 0.659096, 0.666861, 0.666715, 0.668138, 0.668186,
                0.53964,  0.668141, 0.668109, 0.619255, 0.668141, 0.647193, 0.662341, 0.661921, 0.667534, 0.66811,
                0.54692,  0.667558, 0.667297, 0.604361, 0.667564, 0.631676, 0.652518, 0.651781, 0.664541, 0.667311,
                0.551576, 0.664629, 0.663703, 0.592106, 0.664652, 0.615579, 0.638092, 0.637163, 0.656733, 0.663751,
                0.554596, 0.656917, 0.655047, 0.582718, 0.656967, 0.601233, 0.621878, 0.620939, 0.643723, 0.65514,
                0.556574, 0.643984, 0.641397, 0.575854, 0.644055, 0.589658, 0.606642, 0.605821, 0.627796, 0.641522,
                0.557878, 0.628081, 0.625301, 0.570987, 0.628158, 0.580903, 0.593915, 0.593262, 0.611954, 0.625433,
                0.558742, 0.612216, 0.609684, 0.567605, 0.612287, 0.574556, 0.584071, 0.583581, 0.598219, 0.609803,
                0.559316, 0.598435, 0.596364, 0.565285, 0.598493, 0.57008,  0.576828, 0.576475, 0.587333, 0.596461,
                0.559698, 0.587499, 0.58592,  0.563707, 0.587544, 0.56698,  0.571671, 0.571423, 0.579197, 0.585993,
                0.668182, 0.66458,  0.667903, 0.667432, 0.658361, 0.667935, 0.668185, 0.667547, 0.667307, 0.668186,
                0.66803,  0.656815, 0.666091, 0.664171, 0.646084, 0.666251, 0.668096, 0.66459,  0.663738, 0.668113,
                0.666772, 0.643839, 0.66026,  0.655973, 0.630413, 0.660667, 0.667203, 0.656835, 0.655116, 0.667328,
                0.662084, 0.627922, 0.649014, 0.642661, 0.614386, 0.649671, 0.663395, 0.643868, 0.64149,  0.663807,
                0.652065, 0.61207,  0.633798, 0.626647, 0.600233, 0.634582, 0.654454, 0.627954, 0.625399, 0.65525,
                0.637519, 0.598314, 0.617618, 0.610903, 0.588883, 0.618381, 0.640604, 0.612099, 0.609772, 0.641672,
                0.621298, 0.587406, 0.602959, 0.597357, 0.580333, 0.603611, 0.624467, 0.598338, 0.596436, 0.625592,
                0.606134, 0.57925,  0.591004, 0.586675, 0.57415,  0.591515, 0.608935, 0.587425, 0.585974, 0.609946,
                0.593511, 0.573381, 0.581898, 0.578717, 0.569797, 0.582278, 0.595758, 0.579264, 0.578207, 0.596577,
                0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546,  0.57339,  0.572642, 0.586082,
                0.668174, 0.668159, 0.668178, 0.618792, 0.66788,  0.668183, 0.66818,  0.66818,  0.662345, 0.595566,
                0.667915, 0.667737, 0.667963, 0.603963, 0.665981, 0.668052, 0.668006, 0.668007, 0.652525, 0.585315,
                0.66615,  0.665341, 0.6664,   0.591792, 0.659985, 0.666907, 0.666636, 0.66664,  0.638101, 0.577728,
                0.660409, 0.658471, 0.661057, 0.582484, 0.648575, 0.662479, 0.661698, 0.661709, 0.621887, 0.572305,
                0.649254, 0.646247, 0.650314, 0.575687, 0.633281, 0.652764, 0.651396, 0.651414, 0.60665,  0.568515,
                0.634083, 0.630598, 0.635357, 0.57087,  0.617117, 0.638404, 0.636684, 0.636707, 0.593922, 0.565907,
                0.617895, 0.614559, 0.619142, 0.567524, 0.602533, 0.622196, 0.62046,  0.620482, 0.584076, 0.564129,
                0.603195, 0.600379, 0.604265, 0.56523,  0.59067,  0.606921, 0.605404, 0.605423, 0.576832, 0.562925,
                0.591189, 0.588995, 0.592029, 0.56367,  0.581651, 0.594139, 0.59293,  0.592946, 0.571674, 0.562114,
                0.582036, 0.580415, 0.582661, 0.562616, 0.57509,  0.584239, 0.583333, 0.583345, 0.568079, 0.561569,
                0.668139, 0.668063, 0.668139, 0.667082, 0.653793, 0.663397, 0.640434, 0.668175, 0.667092, 0.571849,
                0.667538, 0.666978, 0.667544, 0.663011, 0.639734, 0.654459, 0.624289, 0.667925, 0.663042, 0.5682,
                0.664556, 0.66269,  0.664578, 0.653734, 0.623561, 0.640611, 0.608777, 0.666203, 0.653791, 0.565691,
                0.656765, 0.653146, 0.65681,  0.639656, 0.608128, 0.624474, 0.59563,  0.660545, 0.639731, 0.563983,
                0.643768, 0.638894, 0.643833, 0.62348,  0.595107, 0.608942, 0.585363, 0.649473, 0.623558, 0.562827,
                0.627845, 0.622696, 0.627915, 0.608056, 0.584968, 0.595763, 0.577763, 0.634345, 0.608125, 0.562048,
                0.611999, 0.607362, 0.612063, 0.595049, 0.577477, 0.585464, 0.572329, 0.61815,  0.595104, 0.561524,
                0.598256, 0.594491, 0.598309, 0.584924, 0.572127, 0.577836, 0.568532, 0.603413, 0.584966, 0.561173,
                0.587362, 0.584504, 0.587403, 0.577445, 0.568392, 0.572381, 0.565918, 0.591359, 0.577475, 0.560938,
                0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568, 0.564137, 0.582163, 0.572127, 0.560781,
                0.668102, 0.668132, 0.66388,  0.667456, 0.657447, 0.606385, 0.667634, 0.620685, 0.668185, 0.668187,
                0.667244, 0.667485, 0.655394, 0.664256, 0.644744, 0.59371,  0.664921, 0.6056,   0.668088, 0.668142,
                0.663529, 0.664358, 0.641868, 0.656146, 0.628916, 0.583917, 0.65754,  0.593086, 0.667146, 0.667567,
                0.654712, 0.656356, 0.625799, 0.642901, 0.612988, 0.576717, 0.644878, 0.583449, 0.66321,  0.664664,
                0.640947, 0.643193, 0.610134, 0.626905, 0.599072, 0.571593, 0.629065, 0.57638,  0.654104, 0.656992,
                0.624826, 0.62722,  0.59673,  0.611138, 0.587988, 0.568023, 0.613126, 0.571356, 0.640142, 0.644091,
                0.609258, 0.611426, 0.586197, 0.59755,  0.579676, 0.56557,  0.599186, 0.567859, 0.623984, 0.628198,
                0.596018, 0.597785, 0.578369, 0.586822, 0.573683, 0.563901, 0.588076, 0.565458, 0.608505, 0.612324,
                0.585658, 0.587002, 0.572757, 0.578824, 0.569471, 0.562771, 0.57974,  0.563825, 0.59541,  0.598524,
                0.577977, 0.578955, 0.568828, 0.573079, 0.566562, 0.562011, 0.573728, 0.56272,  0.585197, 0.587567},
            std::vector<T>{0.559698, 0.587499, 0.58592,  0.563707, 0.587544, 0.56698,  0.571671, 0.571423, 0.579197,
                           0.585993, 0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546,  0.57339,
                           0.572642, 0.586082, 0.582036, 0.580415, 0.582661, 0.562616, 0.57509,  0.584239, 0.583333,
                           0.583345, 0.568079, 0.561569, 0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568,
                           0.564137, 0.582163, 0.572127, 0.560781, 0.577977, 0.578955, 0.568828, 0.573079, 0.566562,
                           0.562011, 0.573728, 0.56272,  0.585197, 0.587567},
            std::vector<T>{1.2132,  1.37242, 1.3621,  1.23365, 1.37271, 1.25089, 1.27652, 1.27513, 1.32014, 1.36258,
                           1.34833, 1.26322, 1.29695, 1.284,   1.24985, 1.29853, 1.35913, 1.2862,  1.28197, 1.36315,
                           1.33748, 1.32752, 1.34137, 1.22801, 1.29593, 1.35132, 1.34559, 1.34566, 1.25679, 1.22266,
                           1.32026, 1.30789, 1.32044, 1.27895, 1.24474, 1.25944, 1.23589, 1.33827, 1.27907, 1.21865,
                           1.31284, 1.31868, 1.26086, 1.28443, 1.24866, 1.22491, 1.28812, 1.22855, 1.35744, 1.37287}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::REVERSE,
            TensorIteratorBodyType::LSTM,
            ET,
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 9.13242, 2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457,
                5.55154, 7.24966, 5.1128,  4.25147, 8.34407, 1.4123,  4.49045, 5.12671, 7.62159, 9.18673, 3.49665,
                8.35992, 6.90684, 1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283,  9.88249,
                4.48422, 8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155,  9.94928, 6.88321, 1.33685,
                7.4718,  7.19305, 6.47932, 1.9559,  3.52616, 7.98593, 9.0115,  5.59539, 7.44137, 1.70001, 6.53774,
                8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153, 5.4073,  9.00437,
                8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473, 7.95084, 2.11852, 9.28916,
                1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549, 3.75218, 8.98228, 8.98158, 5.63695,
                3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035, 8.37644, 8.54954, 5.70516, 2.45744, 9.54079,
                1.53504, 8.9785,  6.1691,  4.40962, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                           6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                           6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 10},
            std::vector<T>{
                0.559698, 0.587499, 0.58592,  0.563707, 0.587544, 0.56698,  0.571671, 0.571423, 0.579197, 0.585993,
                0.559316, 0.598435, 0.596364, 0.565285, 0.598493, 0.57008,  0.576828, 0.576475, 0.587333, 0.596461,
                0.558742, 0.612216, 0.609684, 0.567605, 0.612287, 0.574556, 0.584071, 0.583581, 0.598219, 0.609803,
                0.557878, 0.628081, 0.625301, 0.570987, 0.628158, 0.580903, 0.593915, 0.593262, 0.611954, 0.625433,
                0.556574, 0.643984, 0.641397, 0.575854, 0.644055, 0.589658, 0.606642, 0.605821, 0.627796, 0.641522,
                0.554596, 0.656917, 0.655047, 0.582718, 0.656967, 0.601233, 0.621878, 0.620939, 0.643723, 0.65514,
                0.551576, 0.664629, 0.663703, 0.592106, 0.664652, 0.615579, 0.638092, 0.637163, 0.656733, 0.663751,
                0.54692,  0.667558, 0.667297, 0.604361, 0.667564, 0.631676, 0.652518, 0.651781, 0.664541, 0.667311,
                0.53964,  0.668141, 0.668109, 0.619255, 0.668141, 0.647193, 0.662341, 0.661921, 0.667534, 0.66811,
                0.528016, 0.668187, 0.668186, 0.635471, 0.668187, 0.659096, 0.666861, 0.666715, 0.668138, 0.668186,
                0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546,  0.57339,  0.572642, 0.586082,
                0.593511, 0.573381, 0.581898, 0.578717, 0.569797, 0.582278, 0.595758, 0.579264, 0.578207, 0.596577,
                0.606134, 0.57925,  0.591004, 0.586675, 0.57415,  0.591515, 0.608935, 0.587425, 0.585974, 0.609946,
                0.621298, 0.587406, 0.602959, 0.597357, 0.580333, 0.603611, 0.624467, 0.598338, 0.596436, 0.625592,
                0.637519, 0.598314, 0.617618, 0.610903, 0.588883, 0.618381, 0.640604, 0.612099, 0.609772, 0.641672,
                0.652065, 0.61207,  0.633798, 0.626647, 0.600233, 0.634582, 0.654454, 0.627954, 0.625399, 0.65525,
                0.662084, 0.627922, 0.649014, 0.642661, 0.614386, 0.649671, 0.663395, 0.643868, 0.64149,  0.663807,
                0.666772, 0.643839, 0.66026,  0.655973, 0.630413, 0.660667, 0.667203, 0.656835, 0.655116, 0.667328,
                0.66803,  0.656815, 0.666091, 0.664171, 0.646084, 0.666251, 0.668096, 0.66459,  0.663738, 0.668113,
                0.668182, 0.66458,  0.667903, 0.667432, 0.658361, 0.667935, 0.668185, 0.667547, 0.667307, 0.668186,
                0.582036, 0.580415, 0.582661, 0.562616, 0.57509,  0.584239, 0.583333, 0.583345, 0.568079, 0.561569,
                0.591189, 0.588995, 0.592029, 0.56367,  0.581651, 0.594139, 0.59293,  0.592946, 0.571674, 0.562114,
                0.603195, 0.600379, 0.604265, 0.56523,  0.59067,  0.606921, 0.605404, 0.605423, 0.576832, 0.562925,
                0.617895, 0.614559, 0.619142, 0.567524, 0.602533, 0.622196, 0.62046,  0.620482, 0.584076, 0.564129,
                0.634083, 0.630598, 0.635357, 0.57087,  0.617117, 0.638404, 0.636684, 0.636707, 0.593922, 0.565907,
                0.649254, 0.646247, 0.650314, 0.575687, 0.633281, 0.652764, 0.651396, 0.651414, 0.60665,  0.568515,
                0.660409, 0.658471, 0.661057, 0.582484, 0.648575, 0.662479, 0.661698, 0.661709, 0.621887, 0.572305,
                0.66615,  0.665341, 0.6664,   0.591792, 0.659985, 0.666907, 0.666636, 0.66664,  0.638101, 0.577728,
                0.667915, 0.667737, 0.667963, 0.603963, 0.665981, 0.668052, 0.668006, 0.668007, 0.652525, 0.585315,
                0.668174, 0.668159, 0.668178, 0.618792, 0.66788,  0.668183, 0.66818,  0.66818,  0.662345, 0.595566,
                0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568, 0.564137, 0.582163, 0.572127, 0.560781,
                0.587362, 0.584504, 0.587403, 0.577445, 0.568392, 0.572381, 0.565918, 0.591359, 0.577475, 0.560938,
                0.598256, 0.594491, 0.598309, 0.584924, 0.572127, 0.577836, 0.568532, 0.603413, 0.584966, 0.561173,
                0.611999, 0.607362, 0.612063, 0.595049, 0.577477, 0.585464, 0.572329, 0.61815,  0.595104, 0.561524,
                0.627845, 0.622696, 0.627915, 0.608056, 0.584968, 0.595763, 0.577763, 0.634345, 0.608125, 0.562048,
                0.643768, 0.638894, 0.643833, 0.62348,  0.595107, 0.608942, 0.585363, 0.649473, 0.623558, 0.562827,
                0.656765, 0.653146, 0.65681,  0.639656, 0.608128, 0.624474, 0.59563,  0.660545, 0.639731, 0.563983,
                0.664556, 0.66269,  0.664578, 0.653734, 0.623561, 0.640611, 0.608777, 0.666203, 0.653791, 0.565691,
                0.667538, 0.666978, 0.667544, 0.663011, 0.639734, 0.654459, 0.624289, 0.667925, 0.663042, 0.5682,
                0.668139, 0.668063, 0.668139, 0.667082, 0.653793, 0.663397, 0.640434, 0.668175, 0.667092, 0.571849,
                0.577977, 0.578955, 0.568828, 0.573079, 0.566562, 0.562011, 0.573728, 0.56272,  0.585197, 0.587567,
                0.585658, 0.587002, 0.572757, 0.578824, 0.569471, 0.562771, 0.57974,  0.563825, 0.59541,  0.598524,
                0.596018, 0.597785, 0.578369, 0.586822, 0.573683, 0.563901, 0.588076, 0.565458, 0.608505, 0.612324,
                0.609258, 0.611426, 0.586197, 0.59755,  0.579676, 0.56557,  0.599186, 0.567859, 0.623984, 0.628198,
                0.624826, 0.62722,  0.59673,  0.611138, 0.587988, 0.568023, 0.613126, 0.571356, 0.640142, 0.644091,
                0.640947, 0.643193, 0.610134, 0.626905, 0.599072, 0.571593, 0.629065, 0.57638,  0.654104, 0.656992,
                0.654712, 0.656356, 0.625799, 0.642901, 0.612988, 0.576717, 0.644878, 0.583449, 0.66321,  0.664664,
                0.663529, 0.664358, 0.641868, 0.656146, 0.628916, 0.583917, 0.65754,  0.593086, 0.667146, 0.667567,
                0.667244, 0.667485, 0.655394, 0.664256, 0.644744, 0.59371,  0.664921, 0.6056,   0.668088, 0.668142,
                0.668102, 0.668132, 0.66388,  0.667456, 0.657447, 0.606385, 0.667634, 0.620685, 0.668185, 0.668187},
            std::vector<T>{0.559698, 0.587499, 0.58592,  0.563707, 0.587544, 0.56698,  0.571671, 0.571423, 0.579197,
                           0.585993, 0.583768, 0.569262, 0.575267, 0.573003, 0.566785, 0.575539, 0.58546,  0.57339,
                           0.572642, 0.586082, 0.582036, 0.580415, 0.582661, 0.562616, 0.57509,  0.584239, 0.583333,
                           0.583345, 0.568079, 0.561569, 0.579218, 0.577141, 0.579248, 0.572105, 0.565823, 0.568568,
                           0.564137, 0.582163, 0.572127, 0.560781, 0.577977, 0.578955, 0.568828, 0.573079, 0.566562,
                           0.562011, 0.573728, 0.56272,  0.585197, 0.587567},
            std::vector<T>{1.2132,  1.37242, 1.3621,  1.23365, 1.37271, 1.25089, 1.27652, 1.27513, 1.32014, 1.36258,
                           1.34833, 1.26322, 1.29695, 1.284,   1.24985, 1.29853, 1.35913, 1.2862,  1.28197, 1.36315,
                           1.33748, 1.32752, 1.34137, 1.22801, 1.29593, 1.35132, 1.34559, 1.34566, 1.25679, 1.22266,
                           1.32026, 1.30789, 1.32044, 1.27895, 1.24474, 1.25944, 1.23589, 1.33827, 1.27907, 1.21865,
                           1.31284, 1.31868, 1.26086, 1.28443, 1.24866, 1.22491, 1.28812, 1.22855, 1.35744, 1.37287}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::FORWARD,
            TensorIteratorBodyType::GRU,
            ET,
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 9.13242, 2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457,
                5.55154, 7.24966, 5.1128,  4.25147, 8.34407, 1.4123,  4.49045, 5.12671, 7.62159, 9.18673, 3.49665,
                8.35992, 6.90684, 1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283,  9.88249,
                4.48422, 8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155,  9.94928, 6.88321, 1.33685,
                7.4718,  7.19305, 6.47932, 1.9559,  3.52616, 7.98593, 9.0115,  5.59539, 7.44137, 1.70001, 6.53774,
                8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153, 5.4073,  9.00437,
                8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473, 7.95084, 2.11852, 9.28916,
                1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549, 3.75218, 8.98228, 8.98158, 5.63695,
                3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035, 8.37644, 8.54954, 5.70516, 2.45744, 9.54079,
                1.53504, 8.9785,  6.1691,  4.40962, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                           6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<T>{0},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  10},
            std::vector<T>{1,      9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168, 3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 10},
            std::vector<T>{
                0.868724, 6.86548,  6.47683,  1.63923,  6.87664,  2.28849,  3.25364,  3.2015,   4.89637,  6.49477,
                0.781008, 4.78797,  4.52828,  1.29585,  4.79543,  1.72968,  2.37458,  2.33974,  3.47223,  4.54026,
                0.722396, 3.3998,   3.22628,  1.06641,  3.40478,  1.35629,  1.7872,   1.76392,  2.52064,  3.23428,
                0.683233, 2.47224,  2.3563,   0.913096, 2.47557,  1.10679,  1.39472,  1.37917,  1.8848,   2.36164,
                0.657064, 1.85246,  1.77498,  0.810656, 1.85468,  0.94008,  1.13247,  1.12208,  1.45993,  1.77856,
                0.639579, 1.43833,  1.38656,  0.742207, 1.43981,  0.828687, 0.957241, 0.950296, 1.17605,  1.38895,
                0.627896, 1.16161,  1.12702,  0.69647,  1.1626,   0.754255, 0.840153, 0.835513, 0.986357, 1.12861,
                0.620089, 0.976709, 0.953597, 0.66591,  0.977373, 0.704521, 0.761917, 0.758816, 0.859608, 0.954663,
                0.614872, 0.853162, 0.837718, 0.645489, 0.853605, 0.671289, 0.70964,  0.707568, 0.774916, 0.838431,
                0.611387, 0.770609, 0.76029,  0.631845, 0.770905, 0.649083, 0.674709, 0.673325, 0.718326, 0.760766,
                5.95818,  2.75265,  4.02319,  3.53551,  2.24933,  4.08252,  6.36501,  3.61821,  3.45881,  6.51635,
                4.18172,  2.03983,  2.88879,  2.56292,  1.70351,  2.92843,  4.45356,  2.61818,  2.51167,  4.55468,
                2.99471,  1.56352,  2.13079,  1.91305,  1.3388,   2.15728,  3.17635,  1.94997,  1.87881,  3.24392,
                2.20156,  1.24526,  1.6243,   1.47881,  1.09511,  1.642,    2.32293,  1.50349,  1.45593,  2.36808,
                1.67159,  1.03261,  1.28588,  1.18866,  0.932274, 1.2977,   1.75269,  1.20515,  1.17337,  1.78286,
                1.31748,  0.890512, 1.05974,  0.994786, 0.823471, 1.06765,  1.37166,  1.0058,   0.984569, 1.39182,
                1.08086,  0.795566, 0.908644, 0.86524,  0.75077,  0.913924, 1.11707,  0.872601, 0.858414, 1.13054,
                0.922752, 0.732124, 0.807681, 0.778679, 0.702192, 0.81121,  0.946946, 0.783598, 0.774118, 0.955946,
                0.817109, 0.689733, 0.740219, 0.720841, 0.669733, 0.742577, 0.833275, 0.724127, 0.717793, 0.839288,
                0.746518, 0.661408, 0.695142, 0.682194, 0.648044, 0.696717, 0.75732,  0.684389, 0.680157, 0.761339,
                5.54972,  5.17435,  5.69626,  1.4268,   3.98473,  6.07069,  5.85496,  5.8577,   2.51076,  1.22499,
                3.90879,  3.65797,  4.00671,  1.15391,  2.86309,  4.2569,   4.11275,  4.11458,  1.8782,   1.01906,
                2.81234,  2.64475,  2.87777,  0.971563, 2.11362,  3.04494,  2.94862,  2.94985,  1.45553,  0.881462,
                2.07971,  1.96773,  2.12343,  0.849723, 1.61283,  2.23513,  2.17077,  2.17159,  1.1731,   0.789519,
                1.59017,  1.51535,  1.61938,  0.768311, 1.27821,  1.69402,  1.65102,  1.65157,  0.984388, 0.728083,
                1.26307,  1.21307,  1.28259,  0.713913, 1.05462,  1.33246,  1.30373,  1.30409,  0.858293, 0.687033,
                1.0445,   1.0111,   1.05755,  0.677564, 0.905221, 1.09087,  1.07167,  1.07192,  0.774037, 0.659604,
                0.898462, 0.876139, 0.907177, 0.653277, 0.805394, 0.929444, 0.916614, 0.916777, 0.717739, 0.641276,
                0.800878, 0.785962, 0.806701, 0.637048, 0.738691, 0.821579, 0.813007, 0.813116, 0.680121, 0.629029,
                0.735673, 0.725707, 0.739564, 0.626204, 0.694121, 0.749506, 0.743778, 0.743851, 0.654985, 0.620846,
                4.90107,  4.43524,  4.90784,  3.34509,  2.05673,  2.61047,  1.72339,  5.57933,  3.34961,  1.07422,
                3.47537,  3.16411,  3.4799,   2.43568,  1.57482,  1.94482,  1.35209,  3.92858,  2.4387,   0.918317,
                2.52274,  2.31475,  2.52576,  1.82803,  1.25281,  1.50004,  1.10398,  2.82557,  1.83005,  0.814145,
                1.8862,   1.74723,  1.88822,  1.422,    1.03765,  1.20285,  0.938205, 2.08854,  1.42335,  0.744538,
                1.46087,  1.36801,  1.46222,  1.1507,   0.893882, 1.00426,  0.827433, 1.59608,  1.1516,   0.698028,
                1.17667,  1.11463,  1.17758,  0.969422, 0.797818, 0.871573, 0.753417, 1.26702,  0.970024, 0.66695,
                0.986775, 0.945316, 0.987378, 0.848292, 0.733629, 0.782911, 0.703961, 1.04714,  0.848694, 0.646185,
                0.859888, 0.832185, 0.86029,  0.767355, 0.690738, 0.723668, 0.670915, 0.900223, 0.767624, 0.632309,
                0.775103, 0.756592, 0.775372, 0.713274, 0.662079, 0.684083, 0.648834, 0.802055, 0.713453, 0.623038,
                0.718451, 0.706082, 0.718631, 0.677138, 0.64293,  0.657632, 0.634079, 0.73646,  0.677257, 0.616843,
                4.62145,  4.84158,  2.66378,  3.55164,  2.20451,  1.30991,  3.69058,  1.44707,  6.30131,  6.88241,
                3.28853,  3.43562,  1.98044,  2.5737,   1.67356,  1.0758,   2.66654,  1.16745,  4.411,    4.79928,
                2.39789,  2.49618,  1.52384,  1.92025,  1.31879,  0.919376, 1.98228,  0.980615, 3.14791,  3.40736,
                1.80278,  1.86845,  1.21875,  1.48362,  1.08174,  0.814853, 1.52507,  0.855772, 2.30393,  2.47729,
                1.40513,  1.44901,  1.01489,  1.19188,  0.92334,  0.745011, 1.21957,  0.772353, 1.74,     1.85583,
                1.13943,  1.16875,  0.878674, 0.996934, 0.817501, 0.698344, 1.01544,  0.716613, 1.36318,  1.44058,
                0.961889, 0.981481, 0.787656, 0.866675, 0.746781, 0.667161, 0.879041, 0.679369, 1.1114,   1.16312,
                0.843259, 0.85635,  0.726839, 0.779639, 0.699526, 0.646326, 0.787901, 0.654483, 0.943158, 0.977716,
                0.763992, 0.772739, 0.686201, 0.721482, 0.667952, 0.632404, 0.727003, 0.637854, 0.830743, 0.853834,
                0.711027, 0.716871, 0.659048, 0.682622, 0.646854, 0.623101, 0.686311, 0.626743, 0.755629, 0.771058},
            std::vector<T>{0.611387, 0.770609, 0.76029,  0.631845, 0.770905, 0.649083, 0.674709, 0.673325, 0.718326,
                           0.760766, 0.746518, 0.661408, 0.695142, 0.682194, 0.648044, 0.696717, 0.75732,  0.684389,
                           0.680157, 0.761339, 0.735673, 0.725707, 0.739564, 0.626204, 0.694121, 0.749506, 0.743778,
                           0.743851, 0.654985, 0.620846, 0.718451, 0.706082, 0.718631, 0.677138, 0.64293,  0.657632,
                           0.634079, 0.73646,  0.677257, 0.616843, 0.711027, 0.716871, 0.659048, 0.682622, 0.646854,
                           0.623101, 0.686311, 0.626743, 0.755629, 0.771058},
            std::vector<T>{0}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::REVERSE,
            TensorIteratorBodyType::GRU,
            ET,
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  1.39976, 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637,
                8.6232,  8.54902, 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225,  3.99956, 1.08021, 5.54918,
                7.05833, 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
                5.59325, 9.89258, 2.30223, 1.4347,  9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909, 1.00912,
                6.62167, 2.80244, 6.626,   3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902, 6.26823, 9.72608,
                3.73491, 3.8238,  3.03815, 7.05101, 8.0103,  5.61396, 6.53738, 1.41095, 5.0149,  9.71211, 4.23604,
                5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328,  8.2817,  5.12336, 8.98577, 5.80541, 6.19552,
                9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817, 8.57269, 5.99975, 3.42893, 5.38068, 3.48261,
                3.02851, 6.82079, 9.2902,  2.80427, 8.91868, 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755,
                2.49121, 5.52697, 8.08823, 9.13242, 2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457,
                5.55154, 7.24966, 5.1128,  4.25147, 8.34407, 1.4123,  4.49045, 5.12671, 7.62159, 9.18673, 3.49665,
                8.35992, 6.90684, 1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283,  9.88249,
                4.48422, 8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155,  9.94928, 6.88321, 1.33685,
                7.4718,  7.19305, 6.47932, 1.9559,  3.52616, 7.98593, 9.0115,  5.59539, 7.44137, 1.70001, 6.53774,
                8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153, 5.4073,  9.00437,
                8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473, 7.95084, 2.11852, 9.28916,
                1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549, 3.75218, 8.98228, 8.98158, 5.63695,
                3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035, 8.37644, 8.54954, 5.70516, 2.45744, 9.54079,
                1.53504, 8.9785,  6.1691,  4.40962, 10},
            std::vector<T>{1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168,  3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055,  7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
                           7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
                           6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10},
            std::vector<T>{0},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  10},
            std::vector<T>{
                1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,
                3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373,
                8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489,
                4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521,
                2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037,
                6.2595,  6.09321, 6.52544, 9.60882, 3.34881, 3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937,
                1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069,
                4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545, 7.8691,  4.20879, 7.77509, 8.93208,
                1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224,
                7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
                1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976, 2.55051,
                4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
                4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714,
                2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181,
                1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719,
                7.83414, 6.86009, 1.35715, 8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085,
                9.24799, 5.15937, 2.19041, 7.87817, 2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937,
                9.37912, 6.18926, 8.55681, 6.60963, 3.92066, 7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,
                7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202, 9.33716, 3.6506,  2.50256, 1.21691, 5.06801,
                8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
                5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652, 3.1734,  1.21496, 9.69154,
                4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,  8.77422, 3.11741,
                8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766, 8.07546,
                3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
                9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048,
                4.78967, 7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,
                5.86648, 8.73895, 2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565,
                7.35114, 3.1439,  10},
            std::vector<T>{1,      9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
                           8.6168, 3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
                           8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 10},
            std::vector<T>{
                0.611387, 0.770609, 0.76029,  0.631845, 0.770905, 0.649083, 0.674709, 0.673325, 0.718326, 0.760766,
                0.614872, 0.853162, 0.837718, 0.645489, 0.853605, 0.671289, 0.70964,  0.707568, 0.774916, 0.838431,
                0.620089, 0.976709, 0.953597, 0.66591,  0.977373, 0.704521, 0.761917, 0.758816, 0.859608, 0.954663,
                0.627896, 1.16161,  1.12702,  0.69647,  1.1626,   0.754255, 0.840153, 0.835513, 0.986357, 1.12861,
                0.639579, 1.43833,  1.38656,  0.742207, 1.43981,  0.828687, 0.957241, 0.950296, 1.17605,  1.38895,
                0.657064, 1.85246,  1.77498,  0.810656, 1.85468,  0.94008,  1.13247,  1.12208,  1.45993,  1.77856,
                0.683233, 2.47224,  2.3563,   0.913096, 2.47557,  1.10679,  1.39472,  1.37917,  1.8848,   2.36164,
                0.722396, 3.3998,   3.22628,  1.06641,  3.40478,  1.35629,  1.7872,   1.76392,  2.52064,  3.23428,
                0.781008, 4.78797,  4.52828,  1.29585,  4.79543,  1.72968,  2.37458,  2.33974,  3.47223,  4.54026,
                0.868724, 6.86548,  6.47683,  1.63923,  6.87664,  2.28849,  3.25364,  3.2015,   4.89637,  6.49477,
                0.746518, 0.661408, 0.695142, 0.682194, 0.648044, 0.696717, 0.75732,  0.684389, 0.680157, 0.761339,
                0.817109, 0.689733, 0.740219, 0.720841, 0.669733, 0.742577, 0.833275, 0.724127, 0.717793, 0.839288,
                0.922752, 0.732124, 0.807681, 0.778679, 0.702192, 0.81121,  0.946946, 0.783598, 0.774118, 0.955946,
                1.08086,  0.795566, 0.908644, 0.86524,  0.75077,  0.913924, 1.11707,  0.872601, 0.858414, 1.13054,
                1.31748,  0.890512, 1.05974,  0.994786, 0.823471, 1.06765,  1.37166,  1.0058,   0.984569, 1.39182,
                1.67159,  1.03261,  1.28588,  1.18866,  0.932274, 1.2977,   1.75269,  1.20515,  1.17337,  1.78286,
                2.20156,  1.24526,  1.6243,   1.47881,  1.09511,  1.642,    2.32293,  1.50349,  1.45593,  2.36808,
                2.99471,  1.56352,  2.13079,  1.91305,  1.3388,   2.15728,  3.17635,  1.94997,  1.87881,  3.24392,
                4.18172,  2.03983,  2.88879,  2.56292,  1.70351,  2.92843,  4.45356,  2.61818,  2.51167,  4.55468,
                5.95818,  2.75265,  4.02319,  3.53551,  2.24933,  4.08252,  6.36501,  3.61821,  3.45881,  6.51635,
                0.735673, 0.725707, 0.739564, 0.626204, 0.694121, 0.749506, 0.743778, 0.743851, 0.654985, 0.620846,
                0.800878, 0.785962, 0.806701, 0.637048, 0.738691, 0.821579, 0.813007, 0.813116, 0.680121, 0.629029,
                0.898462, 0.876139, 0.907177, 0.653277, 0.805394, 0.929444, 0.916614, 0.916777, 0.717739, 0.641276,
                1.0445,   1.0111,   1.05755,  0.677564, 0.905221, 1.09087,  1.07167,  1.07192,  0.774037, 0.659604,
                1.26307,  1.21307,  1.28259,  0.713913, 1.05462,  1.33246,  1.30373,  1.30409,  0.858293, 0.687033,
                1.59017,  1.51535,  1.61938,  0.768311, 1.27821,  1.69402,  1.65102,  1.65157,  0.984388, 0.728083,
                2.07971,  1.96773,  2.12343,  0.849723, 1.61283,  2.23513,  2.17077,  2.17159,  1.1731,   0.789519,
                2.81234,  2.64475,  2.87777,  0.971563, 2.11362,  3.04494,  2.94862,  2.94985,  1.45553,  0.881462,
                3.90879,  3.65797,  4.00671,  1.15391,  2.86309,  4.2569,   4.11275,  4.11458,  1.8782,   1.01906,
                5.54972,  5.17435,  5.69626,  1.4268,   3.98473,  6.07069,  5.85496,  5.8577,   2.51076,  1.22499,
                0.718451, 0.706082, 0.718631, 0.677138, 0.64293,  0.657632, 0.634079, 0.73646,  0.677257, 0.616843,
                0.775103, 0.756592, 0.775372, 0.713274, 0.662079, 0.684083, 0.648834, 0.802055, 0.713453, 0.623038,
                0.859888, 0.832185, 0.86029,  0.767355, 0.690738, 0.723668, 0.670915, 0.900223, 0.767624, 0.632309,
                0.986775, 0.945316, 0.987378, 0.848292, 0.733629, 0.782911, 0.703961, 1.04714,  0.848694, 0.646185,
                1.17667,  1.11463,  1.17758,  0.969422, 0.797818, 0.871573, 0.753417, 1.26702,  0.970024, 0.66695,
                1.46087,  1.36801,  1.46222,  1.1507,   0.893882, 1.00426,  0.827433, 1.59608,  1.1516,   0.698028,
                1.8862,   1.74723,  1.88822,  1.422,    1.03765,  1.20285,  0.938205, 2.08854,  1.42335,  0.744538,
                2.52274,  2.31475,  2.52576,  1.82803,  1.25281,  1.50004,  1.10398,  2.82557,  1.83005,  0.814145,
                3.47537,  3.16411,  3.4799,   2.43568,  1.57482,  1.94482,  1.35209,  3.92858,  2.4387,   0.918317,
                4.90107,  4.43524,  4.90784,  3.34509,  2.05673,  2.61047,  1.72339,  5.57933,  3.34961,  1.07422,
                0.711027, 0.716871, 0.659048, 0.682622, 0.646854, 0.623101, 0.686311, 0.626743, 0.755629, 0.771058,
                0.763992, 0.772739, 0.686201, 0.721482, 0.667952, 0.632404, 0.727003, 0.637854, 0.830743, 0.853834,
                0.843259, 0.85635,  0.726839, 0.779639, 0.699526, 0.646326, 0.787901, 0.654483, 0.943158, 0.977716,
                0.961889, 0.981481, 0.787656, 0.866675, 0.746781, 0.667161, 0.879041, 0.679369, 1.1114,   1.16312,
                1.13943,  1.16875,  0.878674, 0.996934, 0.817501, 0.698344, 1.01544,  0.716613, 1.36318,  1.44058,
                1.40513,  1.44901,  1.01489,  1.19188,  0.92334,  0.745011, 1.21957,  0.772353, 1.74,     1.85583,
                1.80278,  1.86845,  1.21875,  1.48362,  1.08174,  0.814853, 1.52507,  0.855772, 2.30393,  2.47729,
                2.39789,  2.49618,  1.52384,  1.92025,  1.31879,  0.919376, 1.98228,  0.980615, 3.14791,  3.40736,
                3.28853,  3.43562,  1.98044,  2.5737,   1.67356,  1.0758,   2.66654,  1.16745,  4.411,    4.79928,
                4.62145,  4.84158,  2.66378,  3.55164,  2.20451,  1.30991,  3.69058,  1.44707,  6.30131,  6.88241},
            std::vector<T>{0.611387, 0.770609, 0.76029,  0.631845, 0.770905, 0.649083, 0.674709, 0.673325, 0.718326,
                           0.760766, 0.746518, 0.661408, 0.695142, 0.682194, 0.648044, 0.696717, 0.75732,  0.684389,
                           0.680157, 0.761339, 0.735673, 0.725707, 0.739564, 0.626204, 0.694121, 0.749506, 0.743778,
                           0.743851, 0.654985, 0.620846, 0.718451, 0.706082, 0.718631, 0.677138, 0.64293,  0.657632,
                           0.634079, 0.73646,  0.677257, 0.616843, 0.711027, 0.716871, 0.659048, 0.682622, 0.646854,
                           0.623101, 0.686311, 0.626743, 0.755629, 0.771058},
            std::vector<T>{0}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.f,
            ov::op::RecurrentSequenceDirection::FORWARD,
            TensorIteratorBodyType::RNN,
            ET,
            std::vector<T>{-1,           0.780309,   -0.738585,  -0.920481,  0.652872,    0.0641558,  0.91262,
                           -0.0761474,   0.847476,   -0.252158,  -0.690053,  0.784687,    -0.946421,  -0.416995,
                           -0.202512,    0.614577,   0.254189,   0.81585,    0.112795,    0.679838,   -0.899024,
                           0.61247,      0.861631,   -0.271941,  0.381896,   -0.741371,   0.665373,   -0.363293,
                           0.474405,     0.193539,   -0.272586,  0.589941,   0.396962,    0.643758,   0.798932,
                           0.663799,     0.819915,   0.949827,   0.312239,   0.623979,    -0.794512,  -0.524943,
                           -0.24131,     0.398472,   -0.0301095, -0.169556,  0.277324,    0.51858,    0.115046,
                           0.386911,     0.826518,   -0.950774,  0.674757,   -0.23359,    -0.402458,  -0.987406,
                           -0.124885,    0.475875,   -0.248479,  -0.0135011, -0.971921,   -0.501126,  -0.30573,
                           0.593373,     0.876799,   -0.798956,  0.470805,   0.952874,    0.411772,   0.903529,
                           0.855695,     -0.179938,  -0.32381,   0.538217,   -0.330704,   -0.834627,  0.797116,
                           0.57467,      0.434931,   -0.11202,   0.501784,   0.315797,    -0.278605,  -0.243354,
                           0.299264,     -0.888726,  -0.843481,  -0.816194,  0.61021,     -0.869385,  -0.551361,
                           -0.695734,    0.361182,   -0.748082,  -0.485284,  -0.124065,   -0.780665,  -0.370868,
                           -0.298536,    0.522286,   -0.802298,  0.862921,   -0.347842,   -0.146729,  -0.458638,
                           0.57622,      -0.0933521, -0.833528,  -0.859246,  -0.340501,   -0.61579,   0.675731,
                           -0.876629,    0.108469,   0.141222,   -0.0757794, 0.897658,    -0.310452,  0.768898,
                           -0.13792,     0.98896,    0.601007,   0.883268,   -0.241041,   -0.18915,   -0.426225,
                           -0.0989319,   0.530222,   0.159798,   -0.243754,  0.244787,    0.478345,   0.826766,
                           0.0642072,    0.0356427,  -0.794826,  -0.75666,   0.287393,    -0.108071,  -0.84735,
                           -0.694862,    -0.840984,  0.758743,   -0.677052,  0.845901,    -0.992841,  0.605204,
                           -0.392934,    -0.510492,  0.536422,   0.406964,   0.772353,    0.826283,   -0.549379,
                           -0.157033,    -0.668275,  -0.57833,   0.679797,   0.830901,    0.363183,   -0.181486,
                           -0.555743,    0.6706,     0.553474,   0.474691,   0.717404,    0.945023,   -0.180081,
                           0.194028,     0.476884,   -0.466581,  0.526266,   0.861006,    0.749377,   -0.767673,
                           0.934578,     -0.394025,  0.218032,   -0.664486,  0.716852,    0.452785,   -0.869765,
                           0.0361971,    0.190971,   -0.0841559, 0.184484,   -0.361089,   0.639664,   -0.814522,
                           -0.64626,     -0.558575,  -0.0518135, 0.834904,   0.983071,    0.208429,   0.841115,
                           -0.266728,    -0.984396,  0.310033,   -0.663894,  -0.00708379, 0.581608,   0.635922,
                           -0.266473,    -0.742514,  -0.605766,  -0.958209,  0.267088,    -0.0427639, -0.575115,
                           -0.469043,    -0.622282,  0.77962,    0.432287,   -0.862854,   -0.508723,  0.840711,
                           -0.59019,     -0.0682369, 0.526142,   0.0647325,  0.102044,    -0.529739,  -0.448041,
                           -0.966308,    -0.155126,  -0.906004,  -0.881601,  -0.362032,   -0.113877,  -0.662836,
                           -0.397345,    -0.101194,  -0.0538374, 0.408442,   0.40128,     0.187299,   0.94587,
                           0.324356,     0.75563,    -0.171329,  0.59615,    -0.724044,   -0.477747,  -0.546406,
                           0.064904,     0.389431,   -0.512046,  -0.609801,  0.580285,    -0.18924,   -0.129838,
                           0.252768,     0.357634,   -0.137093,  -0.409645,  0.99457,     -0.60545,   0.115919,
                           -0.0537746,   -0.822487,  -0.37496,   0.197357,   -0.901543,   -0.264034,  -0.743536,
                           -0.948014,    0.464231,   -0.473613,  0.422959,   0.354869,    0.641287,   0.582011,
                           0.21152,      0.00800619, -0.138603,  -0.798317,  -0.0269492,  -0.19921,   0.174343,
                           -0.111682,    -0.532153,  0.268423,   -0.541535,  -0.497098,   0.957141,   -0.106795,
                           -0.838224,    -0.760405,  -0.0744435, 0.556972,   -0.203011,   0.248964,   0.59689,
                           -0.0109004,   -0.925239,  0.438413,   0.386685,   -0.369077,   0.673153,   -0.919203,
                           0.259205,     -0.956505,  0.483536,   -0.206068,  0.0391633,   -0.0715966, 0.34823,
                           0.700705,     -0.3679,    -0.368349,  -0.665279,  0.36909,     0.636464,   -0.634393,
                           -0.931031,    0.0198778,  0.556591,   0.233121,   0.880379,    -0.544078,  0.565815,
                           -0.177247,    0.388592,   -0.498401,  0.0146546,  -0.43808,    -0.562895,  0.847527,
                           0.556404,     -0.481485,  -0.54575,   0.586809,   -0.645919,   -0.411731,  0.634336,
                           -0.107599,    0.699691,   0.879165,   -0.605746,  0.851844,    -0.197677,  -0.0638249,
                           -0.550345,    0.427207,   0.281324,   0.82633,    -0.00911417, -0.523082,  0.360537,
                           0.295952,     0.537166,   0.235453,   0.414191,   0.340562,    -0.0328213, 0.828018,
                           -0.944312,    0.806772,   -0.108092,  0.089091,   -0.960954,   0.725746,   0.269557,
                           -0.000429476, -0.231468,  -0.991745,  0.471178,   -0.496647,   0.943754,   -0.815517,
                           -0.069551,    0.263998,   -0.226304,  -0.684247,  -0.0426104,  0.0763018,  0.903734,
                           0.36846,      -0.0844384, -0.0746106, -0.641445,  0.969822,    0.997518,   0.307509,
                           0.622212,     -0.349354,  -0.876697,  -0.7214,    -0.594663,   -0.919986,  0.409152,
                           -0.603743,    -0.4911,    0.703263,   0.314707,   0.612499,    -0.369318,  0.614946,
                           0.770576,     0.371061,   0.593678,   0.750314,   -0.364852,   0.698688,   0.609751,
                           0.142622,     -0.787519,  0.509953,   0.415293,   -0.640467,   0.701937,   0.649218,
                           0.824335,     0.711544,   -0.57001,   -0.32463,   -0.921129,   -0.52984,   -0.750256,
                           -0.445129,    -0.122558,  0.719862,   -0.354157,  0.975094,    0.930568,   0.390521,
                           0.340562,     -0.927739,  0.570913,   0.0577081,  0.345886,    -0.147266,  -0.920045,
                           0.290715,     0.137354,   0.409865,   0.407486,   -0.548271,   0.969365,   -0.763785,
                           -0.589062,    0.906249,   0.869164,   -0.322404,  0.860601,    -0.792338,  -0.74819,
                           -0.11752,     0.246401,   0.215602,   -0.659965,  -0.334239,   -0.701839,  0.916408,
                           -0.870779,    -0.765881,  -0.0786394, -0.25551,   0.903985,    0.159976,   -0.731893,
                           -0.88472,     0.310355,   0.421346,   -0.190523,  0.320507,    0.689287,   0.976754,
                           0.910255,     0.467333,   -0.411659,  0.410252,   0.00145024,  -0.329416,  0.0472609,
                           0.792444,     0.874022,   -0.108247,  0.452289,   0.613927,    -0.608009,  0.0925679,
                           -0.701885,    -0.625309,  -0.233405,  -0.885755,  0.356572,    0.775295,   -0.312218,
                           -0.485195,    -0.760842,  -0.196276,  -0.326445,  -0.837129,   0.260253,   0.125437,
                           -0.848069,    -0.850426,  1},
            std::vector<T>{-1,        0.0194419,  -0.633291, 0.617539,  0.87557,   -0.940787,  0.569968,  -0.672245,
                           -0.132298, 0.148535,   -0.565955, 0.661358,  -0.40092,  -0.278338,  0.738713,  -0.975645,
                           0.350663,  -0.0375085, 0.954534,  -0.57807,  -0.573083, 0.887977,   -0.347673, 0.972259,
                           -0.125333, 0.930768,   -0.484139, 0.519932,  -0.615546, -0.434102,  0.539075,  -0.983636,
                           -0.29435,  -0.532884,  -0.229788, -0.78175,  -0.185304, -0.189241,  0.540048,  0.68374,
                           -0.213783, -0.0673415, -0.791954, -0.618463, 0.345361,  -0.0507364, 0.603086,  -0.504686,
                           0.482923,  1},
            std::vector<T>{0},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{-1,         -0.913489,  0.693052,  0.019041,  0.0899735, 0.267736,   -0.83394,  0.521807,
                           0.782117,   0.297318,   -0.269702, -0.912514, 0.694362,  -0.6763,    -0.041225, 0.136946,
                           -0.95289,   0.0132674,  -0.716936, 0.821415,  0.94,      -0.545582,  -0.79185,  -0.897178,
                           -0.871876,  -0.752293,  -0.219966, -0.144664, 0.981552,  0.738669,   -0.02916,  0.661658,
                           -0.0956457, 0.187724,   0.0764894, -0.644897, 0.984866,  -0.831317,  0.995369,  -0.687197,
                           0.796943,   0.211748,   0.720581,  0.659591,  -0.45931,  -0.407646,  -0.223375, 0.916773,
                           -0.833119,  -0.0234757, -0.377257, -0.914253, 0.754316,  -0.968665,  0.387126,  -0.781003,
                           -0.481575,  0.10334,    0.376482,  -0.540745, 0.596997,  -0.946251,  -0.802122, -0.393072,
                           0.917111,   0.51311,    0.708101,  0.502501,  -0.457544, 0.603489,   0.889737,  0.809528,
                           -0.698177,  0.980047,   0.614314,  0.577663,  0.877533,  -0.0142305, -0.282326, 0.272626,
                           0.154382,   0.950671,   0.537283,  -0.405515, -0.12969,  -0.90906,   0.442845,  -0.982063,
                           0.761418,   0.346103,   0.061014,  -0.221426, 0.655872,  -0.487476,  0.0797454, -0.686778,
                           0.272147,   0.612127,   -0.390442, 1},
            std::vector<T>{-1,         0.141964,  -0.595054, -0.275782,  -0.0524186, -0.100404,  0.356214,  0.892551,
                           -0.595043,  -0.477837, 0.216629,  -0.672137,  0.0817748,  0.356531,   0.22735,   -0.73296,
                           -0.206045,  -0.286988, 0.61286,   0.287588,   0.220286,   0.251281,   -0.327665, -0.648745,
                           0.792805,   0.644284,  -0.355157, -0.430805,  0.363233,   -0.828104,  -0.650602, 0.406752,
                           -0.0604071, 0.370716,  0.38019,   -0.441156,  0.713,      0.756505,   0.41166,   -0.0277081,
                           0.498859,   -0.673484, -0.801405, -0.900367,  0.436681,   -0.758691,  0.138279,  0.677402,
                           -0.744722,  0.939746,  0.669847,  -0.402517,  -0.970535,  0.92845,    -0.662595, 0.89595,
                           0.169792,   0.574887,  0.805658,  -0.042754,  0.419412,   0.372186,   0.730907,  -0.750522,
                           0.985225,   -0.205298, 0.843882,  0.233976,   0.272515,   -0.194655,  -0.405147, -0.61521,
                           0.276029,   0.837373,  -0.765691, -0.61865,   -0.531337,  -0.0268663, 0.804463,  0.958094,
                           -0.625586,  -0.878096, 0.134272,  -0.0648465, -0.356075,  0.447334,   0.745843,  -0.997586,
                           0.994596,   -0.581395, -0.525029, -0.603188,  0.657781,   0.665195,   -0.472963, 0.3469,
                           -0.658721,  -0.485042, -0.038087, 1},
            std::vector<T>{-1, 0.230334, 0.179311, -0.134689, 0.535764, -0.0883306, 0.945667, 0.757567, -0.164013, 1},
            std::vector<T>{
                -0.470427,  -0.735855, -0.329029,  0.999017,  -0.97258,    -0.832011,  0.845522,  0.894361,
                0.958116,   0.224556,  -0.194886,  0.7597,    -0.979301,   0.857585,   -0.317987, 0.646902,
                0.593991,   0.999861,  0.538864,   0.927538,  0.37339,     -0.72924,   -0.950366, 0.998894,
                -0.345412,  0.999566,  -0.905088,  0.653838,  0.931462,    0.963299,   -0.906984, 0.0686653,
                -0.987239,  -0.536094, 0.796584,   0.99365,   -0.558737,   0.999976,   -0.599586, 0.998812,
                0.983484,   -0.701986, 0.423666,   -0.135422, 0.989586,    0.139235,   0.685596,  -0.461324,
                -0.99599,   0.953192,  -0.966443,  0.971833,  0.924096,    -0.966079,  0.999922,  0.619564,
                0.992519,   -0.955464, -0.780334,  0.996958,  0.754961,    -0.856074,  0.390907,  0.0981389,
                0.119931,   -0.998381, 0.999874,   -0.831976, -0.451887,   0.995945,   -0.999099, -0.0742243,
                0.0827845,  0.612425,  0.999317,   -0.937344, 0.983523,    0.995035,   0.585578,  0.977957,
                -0.43647,   0.388421,  -0.258281,  0.999908,  0.831387,    0.667824,   0.562112,  0.922843,
                0.822575,   -0.242546, 0.926723,   0.993825,  0.934094,    0.43523,    -0.883989, 0.998732,
                0.817433,   -0.981772, 0.0274753,  0.835067,  -0.888153,   0.515512,   -0.535921, 0.959418,
                -0.562229,  -0.987868, 0.792129,   0.475789,  0.514164,    -0.984779,  0.0509315, -0.982143,
                -0.67308,   0.999919,  -0.996151,  -0.260185, 0.199735,    0.993083,   0.969637,  -0.910127,
                -0.675983,  0.70171,   -0.299249,  0.829332,  0.944843,    0.999636,   -0.939607, 0.989802,
                0.988641,   0.905483,  -0.646793,  0.164027,  -0.106558,   0.912668,   0.865034,  0.996346,
                -0.954819,  0.658484,  -0.733437,  0.981117,  0.370026,    0.921197,   -0.488649, 0.0900238,
                0.0720321,  0.992835,  0.585617,   -0.46584,  -0.903143,   0.99996,    -0.356482, -0.749733,
                0.932796,   -0.465316, 0.97494,    0.899907,  -0.67506,    -0.965299,  0.454869,  0.988603,
                -0.982064,  0.828854,  -0.220461,  -0.86623,  -0.339239,   -0.96652,   0.991063,  0.991035,
                0.777575,   0.999398,  0.946364,   0.880981,  -0.998482,   0.547716,   0.999092,  -0.992971,
                0.697291,   0.963563,  -0.891479,  0.300176,  0.364938,    0.775309,   -0.820081, -0.376288,
                0.999946,   0.558757,  0.997203,   -0.866317, -0.999996,   -0.941121,  0.784196,  -0.940529,
                -0.276717,  0.491236,  -0.114034,  -0.801807, 0.497822,    -0.998929,  -0.126009, -0.999082,
                0.681976,   -0.725531, 0.510584,   0.12361,   0.125229,    0.977814,   -0.998011, -0.965556,
                -0.631127,  0.871774,  -0.995246,  0.831005,  0.603614,    -0.976149,  0.723436,  0.005006,
                -0.813208,  0.378399,  0.675123,   0.999891,  -0.91046,    0.734962,   0.983588,  0.29022,
                0.353188,   -0.987676, 0.607138,   0.0411221, -0.694228,   0.70539,    0.932037,  0.733177,
                -0.964759,  0.257687,  0.195126,   -0.995997, 0.998685,    0.826683,   -0.990081, 0.991014,
                -0.950626,  -0.146971, -0.715613,  0.841348,  0.998419,    -0.887543,  0.961327,  0.600526,
                -0.994247,  -0.619601, 0.84072,    -0.738013, -0.698475,   0.999502,   0.881153,  -0.793456,
                0.739883,   0.0180047, 0.4453,     -0.485067, 0.313446,    0.99986,    0.801312,  -0.827691,
                0.933498,   0.999094,  0.803509,   -0.98389,  -0.00203269, 0.846717,   -0.988731, -0.155845,
                0.813561,   -0.821903, 0.876179,   -0.974753, 0.978543,    -0.888744,  0.618244,  0.827802,
                -0.891081,  0.997132,  -0.574761,  -0.133318, 0.51666,     -0.998325,  0.998647,  0.557186,
                0.745226,   0.750499,  -0.151921,  0.471127,  -0.0807336,  0.991118,   0.998363,  -0.834192,
                0.995547,   0.970334,  -0.285624,  0.876872,  -0.89536,    0.233029,   -0.512256, 0.0501049,
                0.914272,   -0.446383, -0.0660111, 0.987471,  -0.293181,   0.0090407,  0.993962,  0.725552,
                0.861327,   0.802724,  0.996225,   -0.357275, 0.692737,    -0.765375,  -0.923606, 0.94725,
                -0.976212,  0.112285,  0.116271,   0.625773,  -0.107807,   -0.991827,  0.616004,  -0.187246,
                -0.546877,  0.598621,  0.984211,   0.834327,  -0.949712,   0.697382,   0.314412,  0.264601,
                -0.0311285, -0.167991, -0.815124,  0.938068,  -0.997105,   -0.0607237, 0.323916,  -0.751497,
                0.967815,   0.488129,  0.992229,   0.909782,  -0.994726,   0.944747,   0.0310377, -0.997291,
                -0.57774,   0.999577,  0.952662,   -0.993977, 0.966995,    0.653885,   0.81589,   -0.00180226,
                0.919955,   0.999967,  -0.388806,  -0.69297,  0.998599,    0.989852,   0.977406,  0.454365,
                -0.613587,  0.96152,   0.668411,   -0.834641, 0.808747,    -0.218147,  0.994641,  0.649985,
                0.983425,   -0.999456, -0.993521,  -0.237065, -0.90937,    0.803391,   -0.959971, -0.966409,
                0.914242,   -0.890865, 0.974014,   -0.926672, -0.0687355,  -0.127353,  0.662279,  -0.589603,
                0.901327,   0.980076,  -0.823804,  -0.997316, 0.998387,    -0.547919,  0.932731,  -0.869742,
                -0.873948,  0.587376,  -0.0521998, 0.796093,  0.814562,    -0.270571,  0.85441,   0.943845,
                0.98825,    0.685685,  -0.451584,  0.0440054, -0.999464,   0.999774,   0.460959,  0.681076,
                -0.324321,  0.967583,  0.654874,   -0.168221, 0.667043,    0.960345,   -0.97207,  -0.595059,
                -0.106839,  0.993147,  0.943661,   0.942445,  -0.939552,   0.971532,   -0.300632, -0.791734,
                0.396844,   -0.757931, 0.995531,   0.657585,  0.997931,    -0.830918,  -0.989057, 0.804422,
                0.851206,   0.947814,  -0.89455,   -0.972667, 0.973761,    -0.978947,  0.71407,   -0.969456,
                -0.0211013, 0.75895,   -0.824819,  0.994166,  0.996015,    -0.911606,  0.992728,  -0.180097,
                0.999886,   -0.970702, -0.859906,  0.384982,  0.399817,    -0.871178,  0.992977,  0.360447,
                -0.310061,  -0.999914, 0.999989,   -0.551683, -0.639379,   0.840487,   -0.977291, 0.950401,
                -0.958736,  -0.796325, 0.997133,   -0.937949, 0.994022,    0.99259,    -0.233032, 0.999401,
                0.996386,   0.496829,  0.983234,   0.972622,  0.999547,    0.0118207,  0.977296,  -0.989754,
                -0.984194,  -0.799701, -0.97941,   0.979603,  0.934784,    -0.947689,  -0.950645, -0.962226,
                0.998866,   -0.990042, -0.547825,  0.689601},
            std::vector<T>{0.926723,  0.993825,  0.934094,  0.43523,   -0.883989, 0.998732,  0.817433,   -0.981772,
                           0.0274753, 0.835067,  0.784196,  -0.940529, -0.276717, 0.491236,  -0.114034,  -0.801807,
                           0.497822,  -0.998929, -0.126009, -0.999082, -0.151921, 0.471127,  -0.0807336, 0.991118,
                           0.998363,  -0.834192, 0.995547,  0.970334,  -0.285624, 0.876872,  0.662279,   -0.589603,
                           0.901327,  0.980076,  -0.823804, -0.997316, 0.998387,  -0.547919, 0.932731,   -0.869742,
                           -0.97941,  0.979603,  0.934784,  -0.947689, -0.950645, -0.962226, 0.998866,   -0.990042,
                           -0.547825, 0.689601},
            std::vector<T>{0}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.f,
            ov::op::RecurrentSequenceDirection::REVERSE,
            TensorIteratorBodyType::RNN,
            ET,
            std::vector<T>{-1,           0.780309,   -0.738585,  -0.920481,  0.652872,    0.0641558,  0.91262,
                           -0.0761474,   0.847476,   -0.252158,  -0.690053,  0.784687,    -0.946421,  -0.416995,
                           -0.202512,    0.614577,   0.254189,   0.81585,    0.112795,    0.679838,   -0.899024,
                           0.61247,      0.861631,   -0.271941,  0.381896,   -0.741371,   0.665373,   -0.363293,
                           0.474405,     0.193539,   -0.272586,  0.589941,   0.396962,    0.643758,   0.798932,
                           0.663799,     0.819915,   0.949827,   0.312239,   0.623979,    -0.794512,  -0.524943,
                           -0.24131,     0.398472,   -0.0301095, -0.169556,  0.277324,    0.51858,    0.115046,
                           0.386911,     0.826518,   -0.950774,  0.674757,   -0.23359,    -0.402458,  -0.987406,
                           -0.124885,    0.475875,   -0.248479,  -0.0135011, -0.971921,   -0.501126,  -0.30573,
                           0.593373,     0.876799,   -0.798956,  0.470805,   0.952874,    0.411772,   0.903529,
                           0.855695,     -0.179938,  -0.32381,   0.538217,   -0.330704,   -0.834627,  0.797116,
                           0.57467,      0.434931,   -0.11202,   0.501784,   0.315797,    -0.278605,  -0.243354,
                           0.299264,     -0.888726,  -0.843481,  -0.816194,  0.61021,     -0.869385,  -0.551361,
                           -0.695734,    0.361182,   -0.748082,  -0.485284,  -0.124065,   -0.780665,  -0.370868,
                           -0.298536,    0.522286,   -0.802298,  0.862921,   -0.347842,   -0.146729,  -0.458638,
                           0.57622,      -0.0933521, -0.833528,  -0.859246,  -0.340501,   -0.61579,   0.675731,
                           -0.876629,    0.108469,   0.141222,   -0.0757794, 0.897658,    -0.310452,  0.768898,
                           -0.13792,     0.98896,    0.601007,   0.883268,   -0.241041,   -0.18915,   -0.426225,
                           -0.0989319,   0.530222,   0.159798,   -0.243754,  0.244787,    0.478345,   0.826766,
                           0.0642072,    0.0356427,  -0.794826,  -0.75666,   0.287393,    -0.108071,  -0.84735,
                           -0.694862,    -0.840984,  0.758743,   -0.677052,  0.845901,    -0.992841,  0.605204,
                           -0.392934,    -0.510492,  0.536422,   0.406964,   0.772353,    0.826283,   -0.549379,
                           -0.157033,    -0.668275,  -0.57833,   0.679797,   0.830901,    0.363183,   -0.181486,
                           -0.555743,    0.6706,     0.553474,   0.474691,   0.717404,    0.945023,   -0.180081,
                           0.194028,     0.476884,   -0.466581,  0.526266,   0.861006,    0.749377,   -0.767673,
                           0.934578,     -0.394025,  0.218032,   -0.664486,  0.716852,    0.452785,   -0.869765,
                           0.0361971,    0.190971,   -0.0841559, 0.184484,   -0.361089,   0.639664,   -0.814522,
                           -0.64626,     -0.558575,  -0.0518135, 0.834904,   0.983071,    0.208429,   0.841115,
                           -0.266728,    -0.984396,  0.310033,   -0.663894,  -0.00708379, 0.581608,   0.635922,
                           -0.266473,    -0.742514,  -0.605766,  -0.958209,  0.267088,    -0.0427639, -0.575115,
                           -0.469043,    -0.622282,  0.77962,    0.432287,   -0.862854,   -0.508723,  0.840711,
                           -0.59019,     -0.0682369, 0.526142,   0.0647325,  0.102044,    -0.529739,  -0.448041,
                           -0.966308,    -0.155126,  -0.906004,  -0.881601,  -0.362032,   -0.113877,  -0.662836,
                           -0.397345,    -0.101194,  -0.0538374, 0.408442,   0.40128,     0.187299,   0.94587,
                           0.324356,     0.75563,    -0.171329,  0.59615,    -0.724044,   -0.477747,  -0.546406,
                           0.064904,     0.389431,   -0.512046,  -0.609801,  0.580285,    -0.18924,   -0.129838,
                           0.252768,     0.357634,   -0.137093,  -0.409645,  0.99457,     -0.60545,   0.115919,
                           -0.0537746,   -0.822487,  -0.37496,   0.197357,   -0.901543,   -0.264034,  -0.743536,
                           -0.948014,    0.464231,   -0.473613,  0.422959,   0.354869,    0.641287,   0.582011,
                           0.21152,      0.00800619, -0.138603,  -0.798317,  -0.0269492,  -0.19921,   0.174343,
                           -0.111682,    -0.532153,  0.268423,   -0.541535,  -0.497098,   0.957141,   -0.106795,
                           -0.838224,    -0.760405,  -0.0744435, 0.556972,   -0.203011,   0.248964,   0.59689,
                           -0.0109004,   -0.925239,  0.438413,   0.386685,   -0.369077,   0.673153,   -0.919203,
                           0.259205,     -0.956505,  0.483536,   -0.206068,  0.0391633,   -0.0715966, 0.34823,
                           0.700705,     -0.3679,    -0.368349,  -0.665279,  0.36909,     0.636464,   -0.634393,
                           -0.931031,    0.0198778,  0.556591,   0.233121,   0.880379,    -0.544078,  0.565815,
                           -0.177247,    0.388592,   -0.498401,  0.0146546,  -0.43808,    -0.562895,  0.847527,
                           0.556404,     -0.481485,  -0.54575,   0.586809,   -0.645919,   -0.411731,  0.634336,
                           -0.107599,    0.699691,   0.879165,   -0.605746,  0.851844,    -0.197677,  -0.0638249,
                           -0.550345,    0.427207,   0.281324,   0.82633,    -0.00911417, -0.523082,  0.360537,
                           0.295952,     0.537166,   0.235453,   0.414191,   0.340562,    -0.0328213, 0.828018,
                           -0.944312,    0.806772,   -0.108092,  0.089091,   -0.960954,   0.725746,   0.269557,
                           -0.000429476, -0.231468,  -0.991745,  0.471178,   -0.496647,   0.943754,   -0.815517,
                           -0.069551,    0.263998,   -0.226304,  -0.684247,  -0.0426104,  0.0763018,  0.903734,
                           0.36846,      -0.0844384, -0.0746106, -0.641445,  0.969822,    0.997518,   0.307509,
                           0.622212,     -0.349354,  -0.876697,  -0.7214,    -0.594663,   -0.919986,  0.409152,
                           -0.603743,    -0.4911,    0.703263,   0.314707,   0.612499,    -0.369318,  0.614946,
                           0.770576,     0.371061,   0.593678,   0.750314,   -0.364852,   0.698688,   0.609751,
                           0.142622,     -0.787519,  0.509953,   0.415293,   -0.640467,   0.701937,   0.649218,
                           0.824335,     0.711544,   -0.57001,   -0.32463,   -0.921129,   -0.52984,   -0.750256,
                           -0.445129,    -0.122558,  0.719862,   -0.354157,  0.975094,    0.930568,   0.390521,
                           0.340562,     -0.927739,  0.570913,   0.0577081,  0.345886,    -0.147266,  -0.920045,
                           0.290715,     0.137354,   0.409865,   0.407486,   -0.548271,   0.969365,   -0.763785,
                           -0.589062,    0.906249,   0.869164,   -0.322404,  0.860601,    -0.792338,  -0.74819,
                           -0.11752,     0.246401,   0.215602,   -0.659965,  -0.334239,   -0.701839,  0.916408,
                           -0.870779,    -0.765881,  -0.0786394, -0.25551,   0.903985,    0.159976,   -0.731893,
                           -0.88472,     0.310355,   0.421346,   -0.190523,  0.320507,    0.689287,   0.976754,
                           0.910255,     0.467333,   -0.411659,  0.410252,   0.00145024,  -0.329416,  0.0472609,
                           0.792444,     0.874022,   -0.108247,  0.452289,   0.613927,    -0.608009,  0.0925679,
                           -0.701885,    -0.625309,  -0.233405,  -0.885755,  0.356572,    0.775295,   -0.312218,
                           -0.485195,    -0.760842,  -0.196276,  -0.326445,  -0.837129,   0.260253,   0.125437,
                           -0.848069,    -0.850426,  1},
            std::vector<T>{-1,        0.0194419,  -0.633291, 0.617539,  0.87557,   -0.940787,  0.569968,  -0.672245,
                           -0.132298, 0.148535,   -0.565955, 0.661358,  -0.40092,  -0.278338,  0.738713,  -0.975645,
                           0.350663,  -0.0375085, 0.954534,  -0.57807,  -0.573083, 0.887977,   -0.347673, 0.972259,
                           -0.125333, 0.930768,   -0.484139, 0.519932,  -0.615546, -0.434102,  0.539075,  -0.983636,
                           -0.29435,  -0.532884,  -0.229788, -0.78175,  -0.185304, -0.189241,  0.540048,  0.68374,
                           -0.213783, -0.0673415, -0.791954, -0.618463, 0.345361,  -0.0507364, 0.603086,  -0.504686,
                           0.482923,  1},
            std::vector<T>{0},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{-1,         -0.913489,  0.693052,  0.019041,  0.0899735, 0.267736,   -0.83394,  0.521807,
                           0.782117,   0.297318,   -0.269702, -0.912514, 0.694362,  -0.6763,    -0.041225, 0.136946,
                           -0.95289,   0.0132674,  -0.716936, 0.821415,  0.94,      -0.545582,  -0.79185,  -0.897178,
                           -0.871876,  -0.752293,  -0.219966, -0.144664, 0.981552,  0.738669,   -0.02916,  0.661658,
                           -0.0956457, 0.187724,   0.0764894, -0.644897, 0.984866,  -0.831317,  0.995369,  -0.687197,
                           0.796943,   0.211748,   0.720581,  0.659591,  -0.45931,  -0.407646,  -0.223375, 0.916773,
                           -0.833119,  -0.0234757, -0.377257, -0.914253, 0.754316,  -0.968665,  0.387126,  -0.781003,
                           -0.481575,  0.10334,    0.376482,  -0.540745, 0.596997,  -0.946251,  -0.802122, -0.393072,
                           0.917111,   0.51311,    0.708101,  0.502501,  -0.457544, 0.603489,   0.889737,  0.809528,
                           -0.698177,  0.980047,   0.614314,  0.577663,  0.877533,  -0.0142305, -0.282326, 0.272626,
                           0.154382,   0.950671,   0.537283,  -0.405515, -0.12969,  -0.90906,   0.442845,  -0.982063,
                           0.761418,   0.346103,   0.061014,  -0.221426, 0.655872,  -0.487476,  0.0797454, -0.686778,
                           0.272147,   0.612127,   -0.390442, 1},
            std::vector<T>{-1,         0.141964,  -0.595054, -0.275782,  -0.0524186, -0.100404,  0.356214,  0.892551,
                           -0.595043,  -0.477837, 0.216629,  -0.672137,  0.0817748,  0.356531,   0.22735,   -0.73296,
                           -0.206045,  -0.286988, 0.61286,   0.287588,   0.220286,   0.251281,   -0.327665, -0.648745,
                           0.792805,   0.644284,  -0.355157, -0.430805,  0.363233,   -0.828104,  -0.650602, 0.406752,
                           -0.0604071, 0.370716,  0.38019,   -0.441156,  0.713,      0.756505,   0.41166,   -0.0277081,
                           0.498859,   -0.673484, -0.801405, -0.900367,  0.436681,   -0.758691,  0.138279,  0.677402,
                           -0.744722,  0.939746,  0.669847,  -0.402517,  -0.970535,  0.92845,    -0.662595, 0.89595,
                           0.169792,   0.574887,  0.805658,  -0.042754,  0.419412,   0.372186,   0.730907,  -0.750522,
                           0.985225,   -0.205298, 0.843882,  0.233976,   0.272515,   -0.194655,  -0.405147, -0.61521,
                           0.276029,   0.837373,  -0.765691, -0.61865,   -0.531337,  -0.0268663, 0.804463,  0.958094,
                           -0.625586,  -0.878096, 0.134272,  -0.0648465, -0.356075,  0.447334,   0.745843,  -0.997586,
                           0.994596,   -0.581395, -0.525029, -0.603188,  0.657781,   0.665195,   -0.472963, 0.3469,
                           -0.658721,  -0.485042, -0.038087, 1},
            std::vector<T>{-1, 0.230334, 0.179311, -0.134689, 0.535764, -0.0883306, 0.945667, 0.757567, -0.164013, 1},
            std::vector<T>{
                0.526906,   -0.997383,   -0.695943,  0.999682,   -0.980027,  -0.898274, 0.995111,   0.0799119,
                0.363935,   -0.0884402,  -0.990447,  0.842608,   0.657827,   -0.362801, 0.295894,   0.222178,
                0.972885,   0.957886,    -0.376478,  0.504993,   0.965053,   -0.229597, -0.946319,  0.999672,
                0.998961,   -0.195694,   0.586567,   -0.58356,   0.631116,   0.416555,  -0.725706,  0.0420496,
                -0.999482,  -0.508284,   0.998452,   -0.989286,  0.958206,   0.99888,   -0.480881,  0.982364,
                0.346879,   -0.687323,   -0.160273,  -0.0172902, -0.303112,  -0.950921, 0.991803,   -0.710375,
                -0.933744,  0.991481,    -0.659493,  0.754693,   0.754852,   0.236133,  0.989564,   0.999994,
                0.684852,   -0.369004,   -0.847966,  0.979346,   0.834702,   0.835757,  -0.997023,  0.951418,
                -0.477717,  0.981288,    0.927471,   0.999848,   0.734415,   0.999904,  -0.991467,  -0.766918,
                0.62936,    0.964863,    -0.857313,  -0.0870588, 0.835937,   0.999409,  0.999204,   0.997886,
                -0.999555,  -0.592204,   0.971622,   0.798724,   -0.568013,  0.283743,  0.828658,   -0.549155,
                0.834866,   -0.133557,   0.920764,   0.999356,   0.694179,   -0.30478,  0.427957,   0.281501,
                0.429332,   -0.936185,   0.347986,   0.950708,   -0.888573,  0.608715,  -0.999921,  0.828499,
                -0.150879,  -0.301067,   -0.976568,  0.999748,   0.284666,   0.983777,  -0.940115,  -0.985955,
                0.544464,   0.998682,    -0.969063,  -0.18267,   0.237068,   0.997719,  -0.0337554, 0.905842,
                -0.878285,  0.309217,    -0.0181193, 0.525607,   0.973771,   0.999497,  -0.995735,  0.998789,
                0.789885,   0.999584,    -0.736026,  -0.435883,  -0.953494,  0.903303,  -0.417712,  0.997552,
                -0.981539,  0.869809,    0.98394,    0.991402,   -0.988794,  0.999331,  -0.158609,  0.780733,
                -0.969231,  0.909109,    0.999747,   -0.381524,  0.972722,   0.994431,  0.630485,   0.472457,
                0.995772,   0.91051,     0.911919,   -0.941698,  0.954069,   -0.988044, 0.992782,   -0.139916,
                -0.566348,  0.763459,    -0.0718403, -0.72653,   0.979029,   -0.995935, 0.999778,   -0.738847,
                0.210184,   -0.737883,   0.988825,   -0.816843,  0.0513971,  -0.839016, 0.988178,   -0.992621,
                0.848743,   -0.998577,   -0.929295,  -0.919254,  -0.43992,   0.93494,   -0.647745,  -0.780525,
                0.918286,   0.992679,    0.912136,   0.383811,   -0.994623,  -0.820734, 0.775965,   0.433662,
                -0.926421,  0.989989,    -0.476612,  -0.854611,  0.473324,   0.263526,  0.410454,   -0.995444,
                -0.979617,  0.971752,    -0.698165,  -0.513943,  0.855178,   -0.725843, -0.954888,  0.940128,
                0.956929,   0.996744,    -0.539351,  0.163227,   0.960576,   -0.520992, -0.779952,  -0.939853,
                -0.248751,  -0.933185,   0.96781,    0.998035,   -0.748558,  -0.422557, 0.652144,   0.289789,
                0.942327,   0.989907,    -0.541705,  -0.967179,  -0.992064,  -0.679435, 0.987373,   0.88219,
                -0.990581,  0.966343,    0.149118,   0.900446,   0.967235,   0.996815,  -0.959944,  0.950417,
                -0.998807,  0.981472,    -0.715279,  0.854894,   -0.575615,  -0.996191, 0.938588,   0.99962,
                0.997776,   0.996625,    -0.993116,  -0.974635,  0.797837,   0.757842,  0.414458,   -0.995602,
                0.997473,   -0.928389,   0.585003,   0.685336,   0.35296,    0.999335,  0.815556,   -0.978755,
                0.977322,   0.862941,    0.844783,   -0.999172,  -0.737575,  0.868368,  0.850968,   -0.355691,
                -0.477411,  0.670716,    0.999938,   -0.985769,  0.753579,   -0.861071, -0.947635,  -0.441339,
                -0.636707,  0.958056,    -0.917965,  -0.888682,  0.887396,   -0.469046, 0.878908,   0.343275,
                -0.953879,  0.983756,    -0.265801,  -0.874482,  0.732147,   0.142205,  0.488677,   0.601925,
                0.0526216,  0.707467,    -0.793197,  0.99486,    -0.851224,  -0.910939, 0.657066,   0.603613,
                0.504114,   -0.988843,   0.968051,   0.487372,   -0.996597,  -0.349508, 0.351675,   0.738722,
                0.784989,   -0.98241,    0.901682,   0.0865038,  -0.847449,  0.575283,  0.329635,   0.999976,
                -0.637486,  -0.843608,   0.551505,   -0.177101,  -0.372926,  0.935283,  -0.990545,  -0.149183,
                -0.491596,  0.541562,    0.996025,   0.472454,   -0.845026,  0.991427,  -0.334852,  0.999497,
                -0.0331139, 0.0179286,   -0.837703,  0.512776,   -0.984419,  0.979792,  -0.974191,  0.925265,
                -0.135773,  0.0270891,   0.996536,   0.999985,   0.979748,   0.998403,  -0.993161,  -0.996728,
                0.638566,   0.991593,    -0.560185,  -0.999493,  0.993987,   0.271173,  0.98406,    0.322271,
                -0.334357,  0.997101,    0.943976,   -0.999868,  0.880896,   0.709814,  0.982917,   -0.995932,
                -0.474997,  0.995407,    0.96453,    -0.753175,  0.651881,   0.526273,  0.902097,   0.992134,
                0.507577,   -0.999034,   -0.996382,  -0.673348,  0.819122,   0.779549,  -0.999686,  0.974422,
                0.880611,   0.6546,      0.6598,     0.96634,    -0.920738,  -0.418959, -0.954179,  0.87176,
                -0.330375,  0.223247,    -0.100151,  -0.310826,  0.93752,    0.996072,  0.883849,   0.902299,
                0.105549,   0.799733,    0.118137,   0.89021,    -0.160378,  -0.831619, -0.0241198, 0.723485,
                0.984892,   0.21006,     -0.707005,  -0.612093,  -0.996712,  0.953598,  0.999635,   -0.958756,
                0.196832,   -0.816948,   -0.822502,  -0.969466,  0.440709,   0.915352,  -0.987622,  -0.756492,
                0.811622,   -0.999958,   0.999982,   -0.47131,   -0.907012,  0.897248,  -0.954296,  0.86897,
                0.92591,    -0.945222,   0.996168,   -0.983258,  0.999693,   -0.883999, -0.800457,  0.18094,
                0.985958,   0.362557,    -0.882676,  -0.598648,  0.887364,   -0.970348, 0.756076,   -0.993787,
                -0.968946,  -0.118565,   -0.636271,  0.998778,   -0.0656388, -0.527903, 0.990153,   0.781603,
                0.999725,   -0.246065,   -0.97279,   0.986471,   0.984443,   -0.70469,  0.701,      0.981588,
                0.982162,   -0.994913,   0.99988,    0.698499,   -0.996202,  0.541067,  -0.990485,  0.844747,
                -0.222405,  -0.209739,   0.91219,    -0.989144,  0.999699,   0.724279,  -0.885552,  0.988889,
                0.58029,    0.759885,    0.99201,    0.818603,   0.873055,   -0.884289, 0.99798,    -0.965469,
                -0.480964,  0.475605,    -0.781967,  0.99447,    0.863794,   -0.861781, 0.891732,   -0.547791,
                0.97225,    -0.00379388, 0.342407,   0.92741},
            std::vector<T>{0.526906,  -0.997383,  -0.695943, 0.999682,  -0.980027, -0.898274, 0.995111,   0.0799119,
                           0.363935,  -0.0884402, -0.888573, 0.608715,  -0.999921, 0.828499,  -0.150879,  -0.301067,
                           -0.976568, 0.999748,   0.284666,  0.983777,  -0.979617, 0.971752,  -0.698165,  -0.513943,
                           0.855178,  -0.725843,  -0.954888, 0.940128,  0.956929,  0.996744,  -0.851224,  -0.910939,
                           0.657066,  0.603613,   0.504114,  -0.988843, 0.968051,  0.487372,  -0.996597,  -0.349508,
                           0.105549,  0.799733,   0.118137,  0.89021,   -0.160378, -0.831619, -0.0241198, 0.723485,
                           0.984892,  0.21006},
            std::vector<T>{0}),
    };
    return params;
}

template <ov::element::Type_t ET>
std::vector<TensorIteratorStaticParams> generateParamsBF16() {
    using T = typename ov::element_type_traits<ET>::value_type;

    std::vector<TensorIteratorStaticParams> params{
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::FORWARD,
            TensorIteratorBodyType::LSTM,
            ET,
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  9.5,     4.5625,  9.5,     9.3125,  6,       2.82812, 9.25,    1.07031,
                6.75,    9.3125,  4.5,     3.65625, 5.375,   2.5,     6.4375,  1.21875, 5.9375,  5.0625,  9.3125,
                8.25,    9.25,    4.3125,  4.5625,  6.46875, 9.625,   1.3125,  2.5625,  4.1875,  2.125,   1.70312,
                2.21875, 7.25,    5.5625,  1.10938, 1.1875,  5.125,   9.5,     9.625,   8.4375,  4,       1.13281,
                5.25,    2.57812, 1.94531, 3.98438, 5.5,     2.17188, 9,       8.25,    5.8125,  4.09375, 3.53125,
                9.4375,  4.1875,  6.25,    9.0625,  8.875,   3.17188, 8.625,   1.21875, 9.125,   9.6875,  5.125,
                4.875,   5.90625, 4.125,   8.125,   6.1875,  3.5625,  2.125,   5.40625, 9.5,     6.375,   3.8125,
                1.14062, 9.5625,  6.3125,  2.96875, 4.875,   3.23438, 8.25,    8.75,    3.84375, 3.125,   9,
                8.3125,  6.1875,  5.875,   2.65625, 2.71875, 8.0625,  6.3125,  6.5,     1.42969, 1.48438, 1.14062,
                4.78125, 1.44531, 7.125,   4.59375, 10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125,
                           8,      8.1875,  7.4375, 9.6875,  8.25,    3.8125,  1.82812, 7.21875, 5.65625, 10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125,
                           8,      8.1875,  7.4375, 9.6875,  8.25,    3.8125,  1.82812, 7.21875, 5.65625, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,     2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,  5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625, 1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375, 10},
            std::vector<T>{
                0.523438, 0.667969, 0.667969, 0.667969, 0.667969, 0.523438, 0.632812, 0.664062, 0.667969, 0.640625,
                0.539062, 0.664062, 0.667969, 0.667969, 0.667969, 0.539062, 0.617188, 0.65625,  0.667969, 0.625,
                0.546875, 0.648438, 0.667969, 0.664062, 0.667969, 0.546875, 0.601562, 0.640625, 0.667969, 0.609375,
                0.546875, 0.632812, 0.664062, 0.65625,  0.664062, 0.546875, 0.585938, 0.625,    0.664062, 0.59375,
                0.554688, 0.617188, 0.65625,  0.640625, 0.648438, 0.554688, 0.578125, 0.609375, 0.65625,  0.585938,
                0.554688, 0.601562, 0.640625, 0.625,    0.640625, 0.554688, 0.570312, 0.59375,  0.640625, 0.578125,
                0.554688, 0.59375,  0.625,    0.609375, 0.625,    0.554688, 0.570312, 0.585938, 0.625,    0.570312,
                0.554688, 0.585938, 0.609375, 0.59375,  0.609375, 0.554688, 0.5625,   0.578125, 0.609375, 0.570312,
                0.554688, 0.570312, 0.59375,  0.585938, 0.59375,  0.554688, 0.5625,   0.570312, 0.59375,  0.5625,
                0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625,   0.570312, 0.585938, 0.5625,
                0.65625,  0.617188, 0.664062, 0.648438, 0.664062, 0.664062, 0.667969, 0.664062, 0.667969, 0.667969,
                0.648438, 0.601562, 0.664062, 0.632812, 0.664062, 0.65625,  0.667969, 0.664062, 0.667969, 0.664062,
                0.632812, 0.585938, 0.648438, 0.617188, 0.648438, 0.648438, 0.664062, 0.648438, 0.667969, 0.65625,
                0.617188, 0.578125, 0.632812, 0.601562, 0.632812, 0.632812, 0.65625,  0.632812, 0.664062, 0.648438,
                0.601562, 0.570312, 0.617188, 0.585938, 0.617188, 0.617188, 0.640625, 0.617188, 0.648438, 0.632812,
                0.585938, 0.570312, 0.601562, 0.578125, 0.601562, 0.601562, 0.625,    0.601562, 0.640625, 0.617188,
                0.578125, 0.5625,   0.585938, 0.570312, 0.585938, 0.585938, 0.609375, 0.585938, 0.625,    0.601562,
                0.570312, 0.5625,   0.578125, 0.570312, 0.578125, 0.578125, 0.59375,  0.578125, 0.609375, 0.585938,
                0.570312, 0.5625,   0.570312, 0.5625,   0.570312, 0.570312, 0.585938, 0.570312, 0.59375,  0.578125,
                0.5625,   0.554688, 0.570312, 0.5625,   0.570312, 0.570312, 0.578125, 0.570312, 0.585938, 0.570312,
                0.667969, 0.667969, 0.664062, 0.667969, 0.667969, 0.648438, 0.667969, 0.667969, 0.65625,  0.5625,
                0.667969, 0.664062, 0.65625,  0.667969, 0.664062, 0.640625, 0.664062, 0.667969, 0.640625, 0.5625,
                0.664062, 0.648438, 0.640625, 0.664062, 0.65625,  0.625,    0.65625,  0.664062, 0.625,    0.554688,
                0.65625,  0.632812, 0.625,    0.65625,  0.648438, 0.609375, 0.640625, 0.664062, 0.609375, 0.554688,
                0.648438, 0.617188, 0.609375, 0.640625, 0.632812, 0.59375,  0.625,    0.648438, 0.59375,  0.554688,
                0.632812, 0.601562, 0.59375,  0.625,    0.617188, 0.585938, 0.609375, 0.632812, 0.585938, 0.554688,
                0.617188, 0.59375,  0.585938, 0.609375, 0.601562, 0.578125, 0.59375,  0.617188, 0.578125, 0.554688,
                0.601562, 0.585938, 0.578125, 0.59375,  0.585938, 0.570312, 0.585938, 0.601562, 0.570312, 0.554688,
                0.585938, 0.570312, 0.570312, 0.585938, 0.578125, 0.570312, 0.578125, 0.585938, 0.570312, 0.554688,
                0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625,   0.570312, 0.578125, 0.5625,   0.554688,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.632812, 0.667969, 0.648438,
                0.664062, 0.667969, 0.667969, 0.664062, 0.664062, 0.664062, 0.664062, 0.617188, 0.667969, 0.632812,
                0.65625,  0.664062, 0.667969, 0.648438, 0.65625,  0.65625,  0.648438, 0.601562, 0.667969, 0.617188,
                0.648438, 0.65625,  0.664062, 0.632812, 0.640625, 0.648438, 0.640625, 0.59375,  0.664062, 0.601562,
                0.632812, 0.640625, 0.648438, 0.617188, 0.625,    0.632812, 0.625,    0.585938, 0.648438, 0.59375,
                0.617188, 0.625,    0.632812, 0.601562, 0.609375, 0.617188, 0.609375, 0.570312, 0.640625, 0.585938,
                0.601562, 0.609375, 0.617188, 0.59375,  0.59375,  0.601562, 0.59375,  0.570312, 0.625,    0.570312,
                0.585938, 0.59375,  0.601562, 0.585938, 0.585938, 0.585938, 0.585938, 0.5625,   0.609375, 0.570312,
                0.578125, 0.585938, 0.59375,  0.570312, 0.578125, 0.578125, 0.578125, 0.5625,   0.59375,  0.5625,
                0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312, 0.570312, 0.5625,   0.585938, 0.5625,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.664062, 0.617188, 0.667969, 0.667969, 0.667969,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.65625,  0.601562, 0.667969, 0.664062, 0.667969,
                0.664062, 0.664062, 0.664062, 0.667969, 0.664062, 0.640625, 0.585938, 0.664062, 0.65625,  0.667969,
                0.65625,  0.65625,  0.65625,  0.664062, 0.65625,  0.625,    0.578125, 0.65625,  0.648438, 0.664062,
                0.648438, 0.648438, 0.640625, 0.65625,  0.648438, 0.609375, 0.570312, 0.640625, 0.632812, 0.65625,
                0.632812, 0.632812, 0.625,    0.640625, 0.632812, 0.59375,  0.570312, 0.625,    0.617188, 0.640625,
                0.617188, 0.617188, 0.609375, 0.625,    0.617188, 0.585938, 0.5625,   0.609375, 0.601562, 0.625,
                0.601562, 0.601562, 0.59375,  0.609375, 0.601562, 0.578125, 0.5625,   0.59375,  0.585938, 0.609375,
                0.585938, 0.585938, 0.585938, 0.59375,  0.585938, 0.570312, 0.5625,   0.585938, 0.578125, 0.59375,
                0.578125, 0.578125, 0.578125, 0.585938, 0.578125, 0.570312, 0.554688, 0.578125, 0.570312, 0.585938},
            std::vector<T>{0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625,   0.570312, 0.585938,
                           0.5625,   0.5625,   0.554688, 0.570312, 0.5625,   0.570312, 0.570312, 0.578125, 0.570312,
                           0.585938, 0.570312, 0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625,   0.570312,
                           0.578125, 0.5625,   0.554688, 0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312,
                           0.570312, 0.5625,   0.585938, 0.5625,   0.578125, 0.578125, 0.578125, 0.585938, 0.578125,
                           0.570312, 0.554688, 0.578125, 0.570312, 0.585938},
            std::vector<T>{1.20312, 1.27344, 1.375,   1.32031, 1.35156, 1.20312, 1.22656, 1.25781, 1.375,   1.23438,
                           1.25,    1.21875, 1.26562, 1.23438, 1.26562, 1.26562, 1.32031, 1.26562, 1.35156, 1.28906,
                           1.34375, 1.27344, 1.25781, 1.32031, 1.28906, 1.24219, 1.28125, 1.34375, 1.24219, 1.21875,
                           1.28906, 1.32031, 1.35156, 1.27344, 1.28125, 1.29688, 1.28125, 1.22656, 1.35156, 1.23438,
                           1.32812, 1.32812, 1.32031, 1.35938, 1.32812, 1.25781, 1.21875, 1.32031, 1.28906, 1.375}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::REVERSE,
            TensorIteratorBodyType::LSTM,
            ET,
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  9.5,     4.5625,  9.5,     9.3125,  6,       2.82812, 9.25,    1.07031,
                6.75,    9.3125,  4.5,     3.65625, 5.375,   2.5,     6.4375,  1.21875, 5.9375,  5.0625,  9.3125,
                8.25,    9.25,    4.3125,  4.5625,  6.46875, 9.625,   1.3125,  2.5625,  4.1875,  2.125,   1.70312,
                2.21875, 7.25,    5.5625,  1.10938, 1.1875,  5.125,   9.5,     9.625,   8.4375,  4,       1.13281,
                5.25,    2.57812, 1.94531, 3.98438, 5.5,     2.17188, 9,       8.25,    5.8125,  4.09375, 3.53125,
                9.4375,  4.1875,  6.25,    9.0625,  8.875,   3.17188, 8.625,   1.21875, 9.125,   9.6875,  5.125,
                4.875,   5.90625, 4.125,   8.125,   6.1875,  3.5625,  2.125,   5.40625, 9.5,     6.375,   3.8125,
                1.14062, 9.5625,  6.3125,  2.96875, 4.875,   3.23438, 8.25,    8.75,    3.84375, 3.125,   9,
                8.3125,  6.1875,  5.875,   2.65625, 2.71875, 8.0625,  6.3125,  6.5,     1.42969, 1.48438, 1.14062,
                4.78125, 1.44531, 7.125,   4.59375, 10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125,
                           8,      8.1875,  7.4375, 9.6875,  8.25,    3.8125,  1.82812, 7.21875, 5.65625, 10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125,
                           8,      8.1875,  7.4375, 9.6875,  8.25,    3.8125,  1.82812, 7.21875, 5.65625, 10},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,     2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,  5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625, 1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375, 10},
            std::vector<T>{
                0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625,   0.570312, 0.585938, 0.5625,
                0.554688, 0.570312, 0.59375,  0.585938, 0.59375,  0.554688, 0.5625,   0.570312, 0.59375,  0.5625,
                0.554688, 0.585938, 0.609375, 0.59375,  0.609375, 0.554688, 0.5625,   0.578125, 0.609375, 0.570312,
                0.554688, 0.59375,  0.625,    0.609375, 0.625,    0.554688, 0.570312, 0.585938, 0.625,    0.570312,
                0.554688, 0.601562, 0.640625, 0.625,    0.640625, 0.554688, 0.570312, 0.59375,  0.640625, 0.578125,
                0.554688, 0.617188, 0.65625,  0.640625, 0.648438, 0.554688, 0.578125, 0.609375, 0.65625,  0.585938,
                0.546875, 0.632812, 0.664062, 0.65625,  0.664062, 0.546875, 0.585938, 0.625,    0.664062, 0.59375,
                0.546875, 0.648438, 0.667969, 0.664062, 0.667969, 0.546875, 0.601562, 0.640625, 0.667969, 0.609375,
                0.539062, 0.664062, 0.667969, 0.667969, 0.667969, 0.539062, 0.617188, 0.65625,  0.667969, 0.625,
                0.523438, 0.667969, 0.667969, 0.667969, 0.667969, 0.523438, 0.632812, 0.664062, 0.667969, 0.640625,
                0.5625,   0.554688, 0.570312, 0.5625,   0.570312, 0.570312, 0.578125, 0.570312, 0.585938, 0.570312,
                0.570312, 0.5625,   0.570312, 0.5625,   0.570312, 0.570312, 0.585938, 0.570312, 0.59375,  0.578125,
                0.570312, 0.5625,   0.578125, 0.570312, 0.578125, 0.578125, 0.59375,  0.578125, 0.609375, 0.585938,
                0.578125, 0.5625,   0.585938, 0.570312, 0.585938, 0.585938, 0.609375, 0.585938, 0.625,    0.601562,
                0.585938, 0.570312, 0.601562, 0.578125, 0.601562, 0.601562, 0.625,    0.601562, 0.640625, 0.617188,
                0.601562, 0.570312, 0.617188, 0.585938, 0.617188, 0.617188, 0.640625, 0.617188, 0.648438, 0.632812,
                0.617188, 0.578125, 0.632812, 0.601562, 0.632812, 0.632812, 0.65625,  0.632812, 0.664062, 0.648438,
                0.632812, 0.585938, 0.648438, 0.617188, 0.648438, 0.648438, 0.664062, 0.648438, 0.667969, 0.65625,
                0.648438, 0.601562, 0.664062, 0.632812, 0.664062, 0.65625,  0.667969, 0.664062, 0.667969, 0.664062,
                0.65625,  0.617188, 0.664062, 0.648438, 0.664062, 0.664062, 0.667969, 0.664062, 0.667969, 0.667969,
                0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625,   0.570312, 0.578125, 0.5625,   0.554688,
                0.585938, 0.570312, 0.570312, 0.585938, 0.578125, 0.570312, 0.578125, 0.585938, 0.570312, 0.554688,
                0.601562, 0.585938, 0.578125, 0.59375,  0.585938, 0.570312, 0.585938, 0.601562, 0.570312, 0.554688,
                0.617188, 0.59375,  0.585938, 0.609375, 0.601562, 0.578125, 0.59375,  0.617188, 0.578125, 0.554688,
                0.632812, 0.601562, 0.59375,  0.625,    0.617188, 0.585938, 0.609375, 0.632812, 0.585938, 0.554688,
                0.648438, 0.617188, 0.609375, 0.640625, 0.632812, 0.59375,  0.625,    0.648438, 0.59375,  0.554688,
                0.65625,  0.632812, 0.625,    0.65625,  0.648438, 0.609375, 0.640625, 0.664062, 0.609375, 0.554688,
                0.664062, 0.648438, 0.640625, 0.664062, 0.65625,  0.625,    0.65625,  0.664062, 0.625,    0.554688,
                0.667969, 0.664062, 0.65625,  0.667969, 0.664062, 0.640625, 0.664062, 0.667969, 0.640625, 0.5625,
                0.667969, 0.667969, 0.664062, 0.667969, 0.667969, 0.648438, 0.667969, 0.667969, 0.65625,  0.5625,
                0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312, 0.570312, 0.5625,   0.585938, 0.5625,
                0.578125, 0.585938, 0.59375,  0.570312, 0.578125, 0.578125, 0.578125, 0.5625,   0.59375,  0.5625,
                0.585938, 0.59375,  0.601562, 0.585938, 0.585938, 0.585938, 0.585938, 0.5625,   0.609375, 0.570312,
                0.601562, 0.609375, 0.617188, 0.59375,  0.59375,  0.601562, 0.59375,  0.570312, 0.625,    0.570312,
                0.617188, 0.625,    0.632812, 0.601562, 0.609375, 0.617188, 0.609375, 0.570312, 0.640625, 0.585938,
                0.632812, 0.640625, 0.648438, 0.617188, 0.625,    0.632812, 0.625,    0.585938, 0.648438, 0.59375,
                0.648438, 0.65625,  0.664062, 0.632812, 0.640625, 0.648438, 0.640625, 0.59375,  0.664062, 0.601562,
                0.65625,  0.664062, 0.667969, 0.648438, 0.65625,  0.65625,  0.648438, 0.601562, 0.667969, 0.617188,
                0.664062, 0.667969, 0.667969, 0.664062, 0.664062, 0.664062, 0.664062, 0.617188, 0.667969, 0.632812,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.632812, 0.667969, 0.648438,
                0.578125, 0.578125, 0.578125, 0.585938, 0.578125, 0.570312, 0.554688, 0.578125, 0.570312, 0.585938,
                0.585938, 0.585938, 0.585938, 0.59375,  0.585938, 0.570312, 0.5625,   0.585938, 0.578125, 0.59375,
                0.601562, 0.601562, 0.59375,  0.609375, 0.601562, 0.578125, 0.5625,   0.59375,  0.585938, 0.609375,
                0.617188, 0.617188, 0.609375, 0.625,    0.617188, 0.585938, 0.5625,   0.609375, 0.601562, 0.625,
                0.632812, 0.632812, 0.625,    0.640625, 0.632812, 0.59375,  0.570312, 0.625,    0.617188, 0.640625,
                0.648438, 0.648438, 0.640625, 0.65625,  0.648438, 0.609375, 0.570312, 0.640625, 0.632812, 0.65625,
                0.65625,  0.65625,  0.65625,  0.664062, 0.65625,  0.625,    0.578125, 0.65625,  0.648438, 0.664062,
                0.664062, 0.664062, 0.664062, 0.667969, 0.664062, 0.640625, 0.585938, 0.664062, 0.65625,  0.667969,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.65625,  0.601562, 0.667969, 0.664062, 0.667969,
                0.667969, 0.667969, 0.667969, 0.667969, 0.667969, 0.664062, 0.617188, 0.667969, 0.667969, 0.667969},
            std::vector<T>{0.554688, 0.570312, 0.585938, 0.578125, 0.585938, 0.554688, 0.5625,   0.570312, 0.585938,
                           0.5625,   0.5625,   0.554688, 0.570312, 0.5625,   0.570312, 0.570312, 0.578125, 0.570312,
                           0.585938, 0.570312, 0.578125, 0.570312, 0.570312, 0.578125, 0.570312, 0.5625,   0.570312,
                           0.578125, 0.5625,   0.554688, 0.570312, 0.578125, 0.585938, 0.570312, 0.570312, 0.570312,
                           0.570312, 0.5625,   0.585938, 0.5625,   0.578125, 0.578125, 0.578125, 0.585938, 0.578125,
                           0.570312, 0.554688, 0.578125, 0.570312, 0.585938},
            std::vector<T>{1.20312, 1.27344, 1.375,   1.32031, 1.35156, 1.20312, 1.22656, 1.25781, 1.375,   1.23438,
                           1.25,    1.21875, 1.26562, 1.23438, 1.26562, 1.26562, 1.32031, 1.26562, 1.35156, 1.28906,
                           1.34375, 1.27344, 1.25781, 1.32031, 1.28906, 1.24219, 1.28125, 1.34375, 1.24219, 1.21875,
                           1.28906, 1.32031, 1.35156, 1.27344, 1.28125, 1.29688, 1.28125, 1.22656, 1.35156, 1.23438,
                           1.32812, 1.32812, 1.32031, 1.35938, 1.32812, 1.25781, 1.21875, 1.32031, 1.28906, 1.375}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::FORWARD,
            TensorIteratorBodyType::GRU,
            ET,
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  9.5,     4.5625,  9.5,     9.3125,  6,       2.82812, 9.25,    1.07031,
                6.75,    9.3125,  4.5,     3.65625, 5.375,   2.5,     6.4375,  1.21875, 5.9375,  5.0625,  9.3125,
                8.25,    9.25,    4.3125,  4.5625,  6.46875, 9.625,   1.3125,  2.5625,  4.1875,  2.125,   1.70312,
                2.21875, 7.25,    5.5625,  1.10938, 1.1875,  5.125,   9.5,     9.625,   8.4375,  4,       1.13281,
                5.25,    2.57812, 1.94531, 3.98438, 5.5,     2.17188, 9,       8.25,    5.8125,  4.09375, 3.53125,
                9.4375,  4.1875,  6.25,    9.0625,  8.875,   3.17188, 8.625,   1.21875, 9.125,   9.6875,  5.125,
                4.875,   5.90625, 4.125,   8.125,   6.1875,  3.5625,  2.125,   5.40625, 9.5,     6.375,   3.8125,
                1.14062, 9.5625,  6.3125,  2.96875, 4.875,   3.23438, 8.25,    8.75,    3.84375, 3.125,   9,
                8.3125,  6.1875,  5.875,   2.65625, 2.71875, 8.0625,  6.3125,  6.5,     1.42969, 1.48438, 1.14062,
                4.78125, 1.44531, 7.125,   4.59375, 10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125,
                           8,      8.1875,  7.4375, 9.6875,  8.25,    3.8125,  1.82812, 7.21875, 5.65625, 10},
            std::vector<T>{0},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   10},
            std::vector<T>{1,     4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,     2.3125,
                           3.125, 1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,  5.84375,
                           8.625, 4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625, 10},
            std::vector<T>{
                0.867188, 3.375,    6.875,    5.1875,   6.4375,   0.867188, 1.64062,  2.65625,  6.875,    1.75,
                0.777344, 2.4375,   4.75,     3.65625,  4.5,      0.777344, 1.29688,  1.96875,  4.75,     1.375,
                0.714844, 1.82812,  3.375,    2.625,    3.1875,   0.714844, 1.0625,   1.51562,  3.375,    1.11719,
                0.675781, 1.42188,  2.4375,   1.95312,  2.3125,   0.675781, 0.910156, 1.21875,  2.4375,   0.945312,
                0.648438, 1.14844,  1.82812,  1.5,      1.75,     0.648438, 0.808594, 1.01562,  1.82812,  0.832031,
                0.632812, 0.964844, 1.42188,  1.20312,  1.375,    0.632812, 0.738281, 0.878906, 1.42188,  0.753906,
                0.621094, 0.839844, 1.14844,  1,        1.11719,  0.621094, 0.691406, 0.785156, 1.14844,  0.699219,
                0.613281, 0.761719, 0.964844, 0.867188, 0.945312, 0.613281, 0.660156, 0.722656, 0.964844, 0.664062,
                0.609375, 0.707031, 0.839844, 0.777344, 0.832031, 0.609375, 0.640625, 0.679688, 0.839844, 0.640625,
                0.605469, 0.671875, 0.761719, 0.714844, 0.753906, 0.605469, 0.625,    0.652344, 0.761719, 0.625,
                2.28125,  1.42188,  3.25,     1.98438,  3.1875,   2.9375,   4.875,    3.25,     6.4375,   4.09375,
                1.71875,  1.14844,  2.34375,  1.53125,  2.3125,   2.15625,  3.4375,   2.34375,  4.5,      2.9375,
                1.34375,  0.964844, 1.76562,  1.21875,  1.75,     1.64062,  2.46875,  1.76562,  3.1875,   2.15625,
                1.09375,  0.839844, 1.375,    1.01562,  1.375,    1.29688,  1.84375,  1.375,    2.3125,   1.64062,
                0.929688, 0.761719, 1.11719,  0.878906, 1.11719,  1.0625,   1.4375,   1.11719,  1.75,     1.29688,
                0.816406, 0.707031, 0.945312, 0.785156, 0.945312, 0.910156, 1.15625,  0.945312, 1.375,    1.0625,
                0.746094, 0.671875, 0.832031, 0.722656, 0.832031, 0.808594, 0.972656, 0.832031, 1.11719,  0.910156,
                0.695312, 0.648438, 0.753906, 0.679688, 0.753906, 0.738281, 0.847656, 0.753906, 0.945312, 0.808594,
                0.664062, 0.632812, 0.699219, 0.652344, 0.699219, 0.691406, 0.761719, 0.699219, 0.832031, 0.738281,
                0.640625, 0.621094, 0.664062, 0.632812, 0.664062, 0.660156, 0.707031, 0.664062, 0.753906, 0.691406,
                5.9375,   3.375,    2.71875,  4.9375,   4,        2.09375,  3.53125,  6.125,    2.21875,  1.03125,
                4.15625,  2.4375,   2,        3.5,      2.875,    1.59375,  2.53125,  4.25,     1.6875,   0.886719,
                2.96875,  1.82812,  1.53125,  2.53125,  2.125,    1.26562,  1.89062,  3.03125,  1.32812,  0.792969,
                2.1875,   1.42188,  1.21875,  1.89062,  1.625,    1.04688,  1.46875,  2.21875,  1.08594,  0.730469,
                1.65625,  1.14844,  1.01562,  1.46875,  1.28125,  0.898438, 1.17969,  1.6875,   0.925781, 0.6875,
                1.3125,   0.964844, 0.878906, 1.17969,  1.05469,  0.800781, 0.988281, 1.32812,  0.816406, 0.65625,
                1.07812,  0.839844, 0.785156, 0.988281, 0.902344, 0.730469, 0.855469, 1.08594,  0.746094, 0.636719,
                0.917969, 0.761719, 0.722656, 0.855469, 0.800781, 0.6875,   0.769531, 0.925781, 0.695312, 0.625,
                0.808594, 0.707031, 0.679688, 0.769531, 0.730469, 0.65625,  0.714844, 0.816406, 0.664062, 0.617188,
                0.738281, 0.671875, 0.652344, 0.714844, 0.6875,   0.636719, 0.675781, 0.746094, 0.640625, 0.609375,
                4.0625,   4.875,    6.375,    3.375,    3.625,    4.1875,   3.4375,   1.70312,  6.5,      2.0625,
                2.90625,  3.4375,   4.4375,   2.4375,   2.59375,  3,        2.46875,  1.34375,  4.5,      1.57812,
                2.125,    2.46875,  3.15625,  1.82812,  1.9375,   2.1875,   1.84375,  1.09375,  3.1875,   1.25,
                1.625,    1.84375,  2.28125,  1.42188,  1.5,      1.65625,  1.4375,   0.929688, 2.3125,   1.03125,
                1.28125,  1.4375,   1.71875,  1.14844,  1.20312,  1.3125,   1.15625,  0.816406, 1.75,     0.886719,
                1.05469,  1.15625,  1.34375,  0.964844, 1,        1.07812,  0.972656, 0.746094, 1.375,    0.792969,
                0.902344, 0.972656, 1.09375,  0.839844, 0.867188, 0.917969, 0.847656, 0.695312, 1.11719,  0.730469,
                0.800781, 0.847656, 0.929688, 0.761719, 0.777344, 0.808594, 0.761719, 0.664062, 0.945312, 0.6875,
                0.730469, 0.761719, 0.816406, 0.707031, 0.714844, 0.738281, 0.707031, 0.640625, 0.832031, 0.65625,
                0.6875,   0.707031, 0.746094, 0.671875, 0.675781, 0.691406, 0.671875, 0.625,    0.753906, 0.636719,
                5.53125,  5.65625,  5.125,    6.65625,  5.6875,   2.71875,  1.42188,  5,        3.96875,  6.875,
                3.875,    3.96875,  3.625,    4.625,    4,        2,        1.14844,  3.53125,  2.84375,  4.75,
                2.78125,  2.84375,  2.59375,  3.28125,  2.875,    1.53125,  0.964844, 2.53125,  2.09375,  3.375,
                2.0625,   2.09375,  1.9375,   2.375,    2.125,    1.21875,  0.839844, 1.89062,  1.59375,  2.4375,
                1.57812,  1.59375,  1.5,      1.78125,  1.625,    1.01562,  0.761719, 1.46875,  1.26562,  1.82812,
                1.25,     1.26562,  1.20312,  1.39062,  1.28125,  0.878906, 0.707031, 1.17969,  1.04688,  1.42188,
                1.03125,  1.04688,  1,        1.125,    1.05469,  0.785156, 0.671875, 0.988281, 0.898438, 1.14844,
                0.886719, 0.898438, 0.867188, 0.949219, 0.902344, 0.722656, 0.648438, 0.855469, 0.800781, 0.964844,
                0.792969, 0.800781, 0.777344, 0.832031, 0.800781, 0.679688, 0.632812, 0.769531, 0.730469, 0.839844,
                0.730469, 0.730469, 0.714844, 0.753906, 0.730469, 0.652344, 0.621094, 0.714844, 0.6875,   0.761719},
            std::vector<T>{0.605469, 0.671875, 0.761719, 0.714844, 0.753906, 0.605469, 0.625,    0.652344, 0.761719,
                           0.625,    0.640625, 0.621094, 0.664062, 0.632812, 0.664062, 0.660156, 0.707031, 0.664062,
                           0.753906, 0.691406, 0.738281, 0.671875, 0.652344, 0.714844, 0.6875,   0.636719, 0.675781,
                           0.746094, 0.640625, 0.609375, 0.6875,   0.707031, 0.746094, 0.671875, 0.675781, 0.691406,
                           0.671875, 0.625,    0.753906, 0.636719, 0.730469, 0.730469, 0.714844, 0.753906, 0.730469,
                           0.652344, 0.621094, 0.714844, 0.6875,   0.761719},
            std::vector<T>{0}),
        TensorIteratorStaticParams(
            std::make_shared<TIStaticInputs>(),
            5,
            10,
            10,
            10,
            0.7f,
            ov::op::RecurrentSequenceDirection::REVERSE,
            TensorIteratorBodyType::GRU,
            ET,
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   1.10938, 2.15625, 1.64062, 3.65625, 9.6875,  4.25,    6.125,   3.46875, 2.82812,
                1.66406, 3.26562, 2.375,   7.6875,  2.45312, 2.75,    9.4375,  6.21875, 4.3125,  9.75,    1.45312,
                8.625,   7.65625, 3.15625, 3.6875,  5.4375,  2.84375, 6.5625,  9.8125,  8.4375,  9,       2.40625,
                7.8125,  1.16406, 6.875,   1.625,   1.35938, 5.375,   8.3125,  6.4375,  7.875,   6.125,   5.09375,
                3.84375, 5.78125, 9.875,   1.98438, 6.1875,  2.3125,  4.40625, 5.5625,  5.9375,  2.9375,  7.6875,
                9.25,    7,       5.15625, 3.375,   2.1875,  1.59375, 7.875,   4.3125,  2.90625, 6.65625, 1.67188,
                2.89062, 1.85938, 7.75,    2.45312, 1.59375, 4.1875,  3.34375, 1.85938, 8.25,    2.28125, 2.73438,
                9.375,   6.75,    6.1875,  5.71875, 8.5,     9.3125,  6.625,   3.375,   3.90625, 1.59375, 7.5625,
                7.625,   5.6875,  7.9375,  7.625,   9.125,   2.48438, 9.375,   7.1875,  1.125,   4.8125,  3.09375,
                7.5625,  6.5625,  7.8125,  9.5,     4.5625,  9.5,     9.3125,  6,       2.82812, 9.25,    1.07031,
                6.75,    9.3125,  4.5,     3.65625, 5.375,   2.5,     6.4375,  1.21875, 5.9375,  5.0625,  9.3125,
                8.25,    9.25,    4.3125,  4.5625,  6.46875, 9.625,   1.3125,  2.5625,  4.1875,  2.125,   1.70312,
                2.21875, 7.25,    5.5625,  1.10938, 1.1875,  5.125,   9.5,     9.625,   8.4375,  4,       1.13281,
                5.25,    2.57812, 1.94531, 3.98438, 5.5,     2.17188, 9,       8.25,    5.8125,  4.09375, 3.53125,
                9.4375,  4.1875,  6.25,    9.0625,  8.875,   3.17188, 8.625,   1.21875, 9.125,   9.6875,  5.125,
                4.875,   5.90625, 4.125,   8.125,   6.1875,  3.5625,  2.125,   5.40625, 9.5,     6.375,   3.8125,
                1.14062, 9.5625,  6.3125,  2.96875, 4.875,   3.23438, 8.25,    8.75,    3.84375, 3.125,   9,
                8.3125,  6.1875,  5.875,   2.65625, 2.71875, 8.0625,  6.3125,  6.5,     1.42969, 1.48438, 1.14062,
                4.78125, 1.44531, 7.125,   4.59375, 10},
            std::vector<T>{1,      4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,
                           3.125,  1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375,
                           8.625,  4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,
                           5.8125, 7.03125, 9.25,   4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125,
                           8,      8.1875,  7.4375, 9.6875,  8.25,    3.8125,  1.82812, 7.21875, 5.65625, 10},
            std::vector<T>{0},
            std::vector<int64_t>{10, 10, 10, 10, 10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   10},
            std::vector<T>{
                1,       4.75,    10,      7.46875, 9.375,   1,       2.15625, 3.71875, 10,      2.3125,  3.125,
                1.82812, 4.5625,  2.67188, 4.5,     4.125,   7,       4.5625,  9.375,   5.84375, 8.625,   4.75,
                3.8125,  7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625,  1.25,    5.8125,  7.03125, 9.25,
                4.75,    5.125,   6,       4.875,   2.25,    9.4375,  2.78125, 8,       8.1875,  7.4375,  9.6875,
                8.25,    3.8125,  1.82812, 7.21875, 5.65625, 8.875,   8.75,    9,       8.4375,  1.76562, 8.4375,
                1.34375, 3.45312, 2.53125, 1.53125, 8.875,   7.03125, 1.88281, 6.3125,  4.78125, 7.03125, 9.625,
                4.6875,  5.8125,  2.78125, 7.21875, 3.59375, 3.84375, 2.28125, 7.1875,  8,       8.5,     4.6875,
                1.16406, 1.30469, 7.75,    6.625,   9.875,   6.9375,  7.71875, 3.6875,  3.53125, 5,       8.125,
                3,       1.92188, 1.65625, 5,       5.21875, 9.125,   1.85938, 3.64062, 9.125,   3.59375, 2.0625,
                2.15625, 5.71875, 1.17188, 1.75,    7.125,   9.25,    2.90625, 9.1875,  3.375,   3.6875,  5.4375,
                6.25,    1.47656, 6.0625,  6.15625, 6.5,     2.3125,  9.625,   6.3125,  3.34375, 7.3125,  3.07812,
                1.92188, 5.8125,  4.71875, 9.5,     7.25,    5.4375,  4.71875, 5.875,   1.45312, 7.875,   5.8125,
                1.40625, 6.96875, 2.25,    5.625,   8.125,   9.5,     1.26562, 6.25,    8.9375,  9.125,   5.875,
                2.23438, 5.03125, 2.25,    9,       8.25,    4.375,   4.5625,  5.84375, 2.48438, 6.875,   9.375,
                4.25,    4.125,   6.125,   7.75,    6.75,    7.53125, 2.125,   8.9375,  7.1875,  6.625,   6.8125,
                7.75,    4.1875,  4.125,   7.875,   3.42188, 4.1875,  9.0625,  7.75,    4.84375, 8.875,   9.625,
                1.10156, 6.96875, 5.46875, 6.59375, 1.66406, 2.03125, 8.0625,  9.5,     1.57812, 5.0625,  4.1875,
                6.1875,  9.5,     4.6875,  4.40625, 3.125,   7.875,   9.125,   7.9375,  6.15625, 3.71875, 1.02344,
                7.9375,  6.5625,  2.375,   3.9375,  6.1875,  5.75,    1.07812, 9,       7.375,   4.1875,  5.25,
                9.125,   7.875,   6.625,   5.1875,  1.14062, 3.40625, 9.375,   8.5,     7.1875,  5.9375,  10,
                1.625,   2.54688, 5.25,    2.21875, 7.6875,  9.375,   2.71875, 7.25,    5.1875,  1.59375, 3.0625,
                7.8125,  5.5625,  7.78125, 2.875,   9.25,    1.4375,  7.375,   5.65625, 2.125,   2.54688, 1.17188,
                4.5625,  1.23438, 1.96875, 1.25,    5.5625,  3.21875, 1.92188, 8.75,    3.59375, 5.84375, 3.07812,
                5.96875, 9.6875,  8.5625,  3.5,     2.125,   3.09375, 3.5,     1.82031, 6.25,    6.125,   9.75,
                4.75,    6.0625,  4.3125,  1.16406, 8.3125,  8.1875,  3.59375, 3.09375, 7.4375,  8.25,    6.5,
                4.5,     4.8125,  8.75,    7.75,    7.71875, 4.84375, 6,       4.84375, 2.21875, 4.25,    1.53906,
                2.375,   2.09375, 9.375,   1.39844, 9.25,    1.96875, 8,       3.03125, 6.5625,  7.40625, 1.32031,
                6.03125, 6.875,   10},
            std::vector<T>{1,     4.75,    10,     7.46875, 9.375,   1,       2.15625, 3.71875, 10,     2.3125,
                           3.125, 1.82812, 4.5625, 2.67188, 4.5,     4.125,   7,       4.5625,  9.375,  5.84375,
                           8.625, 4.75,    3.8125, 7.15625, 5.71875, 2.84375, 5,       8.875,   3.0625, 10},
            std::vector<T>{
                0.605469, 0.671875, 0.761719, 0.714844, 0.753906, 0.605469, 0.625,    0.652344, 0.761719, 0.625,
                0.609375, 0.707031, 0.839844, 0.777344, 0.832031, 0.609375, 0.640625, 0.679688, 0.839844, 0.640625,
                0.613281, 0.761719, 0.964844, 0.867188, 0.945312, 0.613281, 0.660156, 0.722656, 0.964844, 0.664062,
                0.621094, 0.839844, 1.14844,  1,        1.11719,  0.621094, 0.691406, 0.785156, 1.14844,  0.699219,
                0.632812, 0.964844, 1.42188,  1.20312,  1.375,    0.632812, 0.738281, 0.878906, 1.42188,  0.753906,
                0.648438, 1.14844,  1.82812,  1.5,      1.75,     0.648438, 0.808594, 1.01562,  1.82812,  0.832031,
                0.675781, 1.42188,  2.4375,   1.95312,  2.3125,   0.675781, 0.910156, 1.21875,  2.4375,   0.945312,
                0.714844, 1.82812,  3.375,    2.625,    3.1875,   0.714844, 1.0625,   1.51562,  3.375,    1.11719,
                0.777344, 2.4375,   4.75,     3.65625,  4.5,      0.777344, 1.29688,  1.96875,  4.75,     1.375,
                0.867188, 3.375,    6.875,    5.1875,   6.4375,   0.867188, 1.64062,  2.65625,  6.875,    1.75,
                0.640625, 0.621094, 0.664062, 0.632812, 0.664062, 0.660156, 0.707031, 0.664062, 0.753906, 0.691406,
                0.664062, 0.632812, 0.699219, 0.652344, 0.699219, 0.691406, 0.761719, 0.699219, 0.832031, 0.738281,
                0.695312, 0.648438, 0.753906, 0.679688, 0.753906, 0.738281, 0.847656, 0.753906, 0.945312, 0.808594,
                0.746094, 0.671875, 0.832031, 0.722656, 0.832031, 0.808594, 0.972656, 0.832031, 1.11719,  0.910156,
                0.816406, 0.707031, 0.945312, 0.785156, 0.945312, 0.910156, 1.15625,  0.945312, 1.375,    1.0625,
                0.929688, 0.761719, 1.11719,  0.878906, 1.11719,  1.0625,   1.4375,   1.11719,  1.75,     1.29688,
                1.09375,  0.839844, 1.375,    1.01562,  1.375,    1.29688,  1.84375,  1.375,    2.3125,   1.64062,
                1.34375,  0.964844, 1.76562,  1.21875,  1.75,     1.64062,  2.46875,  1.76562,  3.1875,   2.15625,
                1.71875,  1.14844,  2.34375,  1.53125,  2.3125,   2.15625,  3.4375,   2.34375,  4.5,      2.9375,
                2.28125,  1.42188,  3.25,     1.98438,  3.1875,   2.9375,   4.875,    3.25,     6.4375,   4.09375,
                0.738281, 0.671875, 0.652344, 0.714844, 0.6875,   0.636719, 0.675781, 0.746094, 0.640625, 0.609375,
                0.808594, 0.707031, 0.679688, 0.769531, 0.730469, 0.65625,  0.714844, 0.816406, 0.664062, 0.617188,
                0.917969, 0.761719, 0.722656, 0.855469, 0.800781, 0.6875,   0.769531, 0.925781, 0.695312, 0.625,
                1.07812,  0.839844, 0.785156, 0.988281, 0.902344, 0.730469, 0.855469, 1.08594,  0.746094, 0.636719,
                1.3125,   0.964844, 0.878906, 1.17969,  1.05469,  0.800781, 0.988281, 1.32812,  0.816406, 0.65625,
                1.65625,  1.14844,  1.01562,  1.46875,  1.28125,  0.898438, 1.17969,  1.6875,   0.925781, 0.6875,
                2.1875,   1.42188,  1.21875,  1.89062,  1.625,    1.04688,  1.46875,  2.21875,  1.08594,  0.730469,
                2.96875,  1.82812,  1.53125,  2.53125,  2.125,    1.26562,  1.89062,  3.03125,  1.32812,  0.792969,
                4.15625,  2.4375,   2,        3.5,      2.875,    1.59375,  2.53125,  4.25,     1.6875,   0.886719,
                5.9375,   3.375,    2.71875,  4.9375,   4,        2.09375,  3.53125,  6.125,    2.21875,  1.03125,
                0.6875,   0.707031, 0.746094, 0.671875, 0.675781, 0.691406, 0.671875, 0.625,    0.753906, 0.636719,
                0.730469, 0.761719, 0.816406, 0.707031, 0.714844, 0.738281, 0.707031, 0.640625, 0.832031, 0.65625,
                0.800781, 0.847656, 0.929688, 0.761719, 0.777344, 0.808594, 0.761719, 0.664062, 0.945312, 0.6875,
                0.902344, 0.972656, 1.09375,  0.839844, 0.867188, 0.917969, 0.847656, 0.695312, 1.11719,  0.730469,
                1.05469,  1.15625,  1.34375,  0.964844, 1,        1.07812,  0.972656, 0.746094, 1.375,    0.792969,
                1.28125,  1.4375,   1.71875,  1.14844,  1.20312,  1.3125,   1.15625,  0.816406, 1.75,     0.886719,
                1.625,    1.84375,  2.28125,  1.42188,  1.5,      1.65625,  1.4375,   0.929688, 2.3125,   1.03125,
                2.125,    2.46875,  3.15625,  1.82812,  1.9375,   2.1875,   1.84375,  1.09375,  3.1875,   1.25,
                2.90625,  3.4375,   4.4375,   2.4375,   2.59375,  3,        2.46875,  1.34375,  4.5,      1.57812,
                4.0625,   4.875,    6.375,    3.375,    3.625,    4.1875,   3.4375,   1.70312,  6.5,      2.0625,
                0.730469, 0.730469, 0.714844, 0.753906, 0.730469, 0.652344, 0.621094, 0.714844, 0.6875,   0.761719,
                0.792969, 0.800781, 0.777344, 0.832031, 0.800781, 0.679688, 0.632812, 0.769531, 0.730469, 0.839844,
                0.886719, 0.898438, 0.867188, 0.949219, 0.902344, 0.722656, 0.648438, 0.855469, 0.800781, 0.964844,
                1.03125,  1.04688,  1,        1.125,    1.05469,  0.785156, 0.671875, 0.988281, 0.898438, 1.14844,
                1.25,     1.26562,  1.20312,  1.39062,  1.28125,  0.878906, 0.707031, 1.17969,  1.04688,  1.42188,
                1.57812,  1.59375,  1.5,      1.78125,  1.625,    1.01562,  0.761719, 1.46875,  1.26562,  1.82812,
                2.0625,   2.09375,  1.9375,   2.375,    2.125,    1.21875,  0.839844, 1.89062,  1.59375,  2.4375,
                2.78125,  2.84375,  2.59375,  3.28125,  2.875,    1.53125,  0.964844, 2.53125,  2.09375,  3.375,
                3.875,    3.96875,  3.625,    4.625,    4,        2,        1.14844,  3.53125,  2.84375,  4.75,
                5.53125,  5.65625,  5.125,    6.65625,  5.6875,   2.71875,  1.42188,  5,        3.96875,  6.875},
            std::vector<T>{0.605469, 0.671875, 0.761719, 0.714844, 0.753906, 0.605469, 0.625,    0.652344, 0.761719,
                           0.625,    0.640625, 0.621094, 0.664062, 0.632812, 0.664062, 0.660156, 0.707031, 0.664062,
                           0.753906, 0.691406, 0.738281, 0.671875, 0.652344, 0.714844, 0.6875,   0.636719, 0.675781,
                           0.746094, 0.640625, 0.609375, 0.6875,   0.707031, 0.746094, 0.671875, 0.675781, 0.691406,
                           0.671875, 0.625,    0.753906, 0.636719, 0.730469, 0.730469, 0.714844, 0.753906, 0.730469,
                           0.652344, 0.621094, 0.714844, 0.6875,   0.761719},
            std::vector<T>{0}),
    };
    return params;
}

std::vector<TensorIteratorStaticParams> generateCombinedParams() {
    const std::vector<std::vector<TensorIteratorStaticParams>> generatedParams{
        generateParams<ov::element::Type_t::f64>(),
        generateParams<ov::element::Type_t::f32>(),
        generateParams<ov::element::Type_t::f16>(),
        generateParamsBF16<ov::element::Type_t::bf16>(),
    };
    std::vector<TensorIteratorStaticParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_TensorIterator_With_Hardcoded_Refs,
                         ReferenceTILayerStaticTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceTILayerStaticTest::getTestCaseName);
}  // namespace
