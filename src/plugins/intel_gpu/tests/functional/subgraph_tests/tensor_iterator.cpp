// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include "shared_test_classes/base/utils/generate_inputs.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

/*
*   Generate TensorIterator with LSTMCell
*   @param ngPrc precision of model
*   @param initShape initial shape {N, L(sequence length), I}
*   @param N batch size
*   @param I input size
*   @param H hidden layer
*/
static std::shared_ptr<ov::Model> makeTIwithLSTMcell(ov::element::Type_t ngPRC, ov::PartialShape initShape,
                                                        size_t N, size_t I, size_t H, size_t sequence_axis,
                                                        ngraph::op::RecurrentSequenceDirection seq_direction) {
    auto SENT = std::make_shared<ov::op::v0::Parameter>(ngPRC, initShape);
    SENT->set_friendly_name("SENT");

    // initial_hidden_state
    auto H_init = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, H});
    H_init->set_friendly_name("H_init");
    // initial_cell_state
    auto C_init = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, H});
    C_init->set_friendly_name("C_init");

    auto H_t = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, H});
    H_t->set_friendly_name("H_t");
    auto C_t = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, H});
    C_t->set_friendly_name("C_t");

    // Body
    // input data
    auto X = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, I});
    X->set_friendly_name("X");

    // the weights for matrix multiplication, gate order: fico
    std::vector<uint64_t> dataW(4 * H * I, 0);
    auto W_body = std::make_shared<ov::op::v0::Constant>(ngPRC, ov::Shape{4 * H, I}, dataW);
    W_body->set_friendly_name("W_body");

    // the recurrence weights for matrix multiplication, gate order: fico
    std::vector<uint64_t> dataR(4 * H * H, 0);
    auto R_body = std::make_shared<ov::op::v0::Constant>(ngPRC, ov::Shape{4 * H, H}, dataR);
    R_body->set_friendly_name("R_body");

    std::vector<uint64_t> inShape = {N, H};
    auto constantH = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, inShape);
    constantH->set_friendly_name("constantH");

    inShape = {N, I};
    auto constantX = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, inShape);
    constantX->set_friendly_name("constantX");

    auto LSTM_cell =
        std::make_shared<ov::op::v4::LSTMCell>(std::make_shared<ov::op::v1::Reshape>(X, constantX, false),
                                               std::make_shared<ov::op::v1::Reshape>(H_t, constantH, false),
                                               std::make_shared<ov::op::v1::Reshape>(C_t, constantH, false),
                                               W_body,
                                               R_body,
                                               H);
    LSTM_cell->set_friendly_name("LSTM_cell");

    inShape = {N, 1, H};
    auto constantHo = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, inShape);
    constantHo->set_friendly_name("constantHo");

    auto H_o = std::make_shared<ov::op::v1::Reshape>(LSTM_cell->output(0), constantHo, false);
    H_o->set_friendly_name("H_o_reshape");
    auto C_o = std::make_shared<ov::op::v1::Reshape>(LSTM_cell->output(1), constantHo, false);
    C_o->set_friendly_name("C_o_reshape");
    auto body = std::make_shared<ov::Model>(ov::OutputVector{H_o, C_o}, ov::ParameterVector{X, H_t, C_t});
    body->set_friendly_name("body");

    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_friendly_name("tensor_iterator");
    tensor_iterator->set_body(body);
    // H_t is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    // Set PortMap
    if (seq_direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
        tensor_iterator->set_sliced_input(X, SENT, 0, 1, 1, -1, sequence_axis);
    } else if (seq_direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
        tensor_iterator->set_sliced_input(X, SENT, -1, -1, 1, 0, sequence_axis);
    } else {
        OPENVINO_THROW("Bidirectional case is not supported.");
    }

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results =
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(out0), std::make_shared<ov::op::v0::Result>(out1)};
    auto fn_ptr = std::make_shared<ov::Model>(results, ov::ParameterVector{SENT, H_init, C_init});
    fn_ptr->set_friendly_name("TIwithLSTMcell");
    return fn_ptr;
}

/*
*   Generate LSTMSequence
*   @param ngPrc precision of model
*   @param initShape initial shape {N, L(sequence length), I}
*   @param N batch size
*   @param I input size
*   @param H hidden layer
*/
static std::shared_ptr<ov::Model> makeLSTMSequence(ov::element::Type_t ngPRC, ov::PartialShape initShape,
                                                        size_t N, size_t I, size_t H, size_t sequence_axis,
                                                        ngraph::op::RecurrentSequenceDirection seq_direction) {
    auto X = std::make_shared<ov::op::v0::Parameter>(ngPRC, initShape);
    auto Y = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, H});
    auto Z = std::make_shared<ov::op::v0::Parameter>(ngPRC, ov::Shape{N, 1, H});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
    auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
    auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

    auto w_val = std::vector<float>(4 * H * I, 0);
    auto r_val = std::vector<float>(4 * H * H, 0);
    auto b_val = std::vector<float>(4 * H, 0);
    auto W = ov::op::v0::Constant::create(ngPRC, ov::Shape{N, 4 * H, I}, w_val);
    auto R = ov::op::v0::Constant::create(ngPRC, ov::Shape{N, 4 * H, H}, r_val);
    auto B = ov::op::v0::Constant::create(ngPRC, ov::Shape{N, 4 * H}, b_val);

    auto rnn_sequence = std::make_shared<ov::op::v5::LSTMSequence>(X,
                                                                Y,
                                                                Z,
                                                                seq_lengths,
                                                                W,
                                                                R,
                                                                B,
                                                                128,
                                                                seq_direction);
    auto Y_out = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(0));
    auto Ho = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(1));
    auto Co = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(2));
    Y_out->set_friendly_name("Y_out");
    Ho->set_friendly_name("Ho");
    Co->set_friendly_name("Co");

    auto fn_ptr = std::make_shared<ov::Model>(ov::NodeVector{Y_out, Ho, Co}, ov::ParameterVector{X, Y, Z});
    fn_ptr->set_friendly_name("LSTMSequence");
    return fn_ptr;
}

enum class LSTMType {
    LSTMCell = 0,
    LSTMSequence = 1
};

using DynamicTensorIteratorParams = typename std::tuple<
        LSTMType,                               // LSTM type (LSTMCell, LSTMSequence)
        InputShape,                             // input shapes (N[batch], L[seq_length], I[input_size])
        int32_t,                                // hidden size
        ngraph::op::RecurrentSequenceDirection, // sequence direction
        std::string,                            // device name
        InferenceEngine::Precision,             // precision
        ov::AnyMap                              // configuration
        >;

/**
 * Test case with Dynamic SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class DynamicTensorIteratorTest : public testing::WithParamInterface<DynamicTensorIteratorParams>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicTensorIteratorParams> &obj) {
        LSTMType type;
        InputShape data_shapes;
        int32_t hidden_size;
        ngraph::op::RecurrentSequenceDirection seq_direction;
        std::string target_device;
        InferenceEngine::Precision data_precision;
        ov::Any configuration;
        std::tie(type, data_shapes,
                    hidden_size,
                    seq_direction,
                    target_device,
                    data_precision,
                    configuration) = obj.param;
        std::ostringstream result;
        result << "TestType=" << (type == LSTMType::LSTMCell? "LSTMCell" : "LSTMSequence") << "_";
        result << "IS=(";
        result << ov::test::utils::partialShape2str({data_shapes.first}) << "_";
        result << ov::test::utils::vec2str(data_shapes.second) << "_";
        result << ")_";
        result << "hidden_size=" << hidden_size << "_";
        result << "direction=" << seq_direction << "_";
        result << "netPRC=" << data_precision << "_";
        result << "targetDevice=" << target_device << "_";
        return result.str();
    }

private:
    InputShape data_shapes;
    ngraph::op::RecurrentSequenceDirection seq_direction;
    InferenceEngine::Precision data_prc;
    size_t hidden_size;
    size_t batch_size;
    size_t input_size;
    LSTMType type;

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        ov::AnyMap configuration_new;
        std::tie(type, data_shapes,
                    hidden_size,
                    seq_direction,
                    targetDevice,
                    data_prc,
                    configuration_new) = GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_prc);
        if (targetDevice == ov::test::utils::DEVICE_GPU) {
            configuration = configuration_new;
        }


        size_t sequence_axis = 1;
        auto init_shape = data_shapes.first;
        init_input_shapes({data_shapes});
        batch_size = static_cast<size_t>(init_shape[0].get_length());
        input_size = static_cast<size_t>(init_shape[init_shape.size()-1].get_length());
        if (type == LSTMType::LSTMCell)
            function = makeTIwithLSTMcell(ngPrc, init_shape, batch_size, input_size, hidden_size, sequence_axis, seq_direction);
        else
            function = makeLSTMSequence(ngPrc, init_shape, batch_size, input_size, hidden_size, sequence_axis, seq_direction);
    }

     void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        ov::Shape default_shape{batch_size, 1, hidden_size};
        auto inputMap = ov::test::utils::getInputMap();
        auto itTargetShape = targetInputStaticShapes.begin();
        for (const auto &param : function->get_parameters()) {
            std::shared_ptr<ov::Node> inputNode = param;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto &node : param->get_output_target_inputs(i)) {
                    std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                    auto it = inputMap.find(nodePtr->get_type_info());
                    ASSERT_NE(it, inputMap.end());
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (itTargetShape != targetInputStaticShapes.end()) {
                            if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                                inputs.insert({param, it->second(nodePtr, port, param->get_element_type(), *itTargetShape)});
                                break;
                            }
                        } else {
                            inputs.insert({param, it->second(nodePtr, port, param->get_element_type(), default_shape)});
                        }
                    }
                }
            }
            if (itTargetShape != targetInputStaticShapes.end())
                itTargetShape++;
        }
    }
};


TEST_P(DynamicTensorIteratorTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

std::vector<LSTMType> lstm_types = {
    LSTMType::LSTMCell, LSTMType::LSTMSequence
};

std::vector<InputShape> input_shapes = {
    InputShape(ov::PartialShape({1, -1, 512}), {{1, 30, 512}, {1, 10, 512}, {1, 5, 512}})
};

std::vector<int32_t> hidden_sizes = {
    128
};

ov::AnyMap net_configuration = {
    {GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, PluginConfigParams::NO}
};

std::vector<InferenceEngine::Precision> net_precision = {
    InferenceEngine::Precision::FP32,
};

std::vector<ngraph::op::RecurrentSequenceDirection> reccurent_sequence_direction = {
    ngraph::op::RecurrentSequenceDirection::FORWARD,
    ngraph::op::RecurrentSequenceDirection::REVERSE,
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicTensorIterator_LSTMCell, DynamicTensorIteratorTest,
                        testing::Combine(
                        /* lstm_type */ testing::ValuesIn({LSTMType::LSTMCell}),
                        /* data_shape */ testing::ValuesIn(input_shapes),
                        /* hidden_size */ testing::ValuesIn(hidden_sizes),
                        /* direction */ testing::ValuesIn(reccurent_sequence_direction),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* data_prc */ testing::ValuesIn(net_precision),
                        /* configuration */ testing::Values<ov::AnyMap>(net_configuration)),
                        DynamicTensorIteratorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicTensorIterator_LSTMSequence, DynamicTensorIteratorTest,
                        testing::Combine(
                        /* lstm_type */ testing::ValuesIn({LSTMType::LSTMSequence}),
                        /* data_shape */ testing::ValuesIn(input_shapes),
                        /* hidden_size */ testing::ValuesIn(hidden_sizes),
                        /* direction */ testing::ValuesIn(reccurent_sequence_direction),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* data_prc */ testing::ValuesIn(net_precision),
                        /* configuration */ testing::Values<ov::AnyMap>(net_configuration)),
                        DynamicTensorIteratorTest::getTestCaseName);
} // namespace GPULayerTestsDefinitions
