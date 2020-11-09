// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <ie_plugin_config.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/common_optimizations/low_latency.hpp"

#include "subgraph_tests/basic_lstm.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string Basic_LSTM_S::getTestCaseName(testing::TestParamInfo<basicLstmParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void Basic_LSTM_S::SetUp() {
    threshold = 0.1f;

    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { {1, 490} });

    hidden_size = 118;
    const size_t batch_size = 1;

    outPrc = InferenceEngine::Precision::FP32;

    //Reshape_1 [1,490] -> [1, 10, 49]
    std::vector<uint64_t> outFormShapes1 = { batch_size, 10, 49 };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{3}, outFormShapes1);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

    auto reshape1_shape = reshape1->output(0).get_shape();
    auto H_init = ngraph::builder::makeConstant<float>(ngPrc, { batch_size, hidden_size }, {}, true);
    auto C_init = ngraph::builder::makeConstant<float>(ngPrc, { batch_size, hidden_size }, {}, true);
    hidden_memory_init = std::static_pointer_cast<ngraph::opset1::Constant>(H_init)->cast_vector<float>();
    cell_memory_init = std::static_pointer_cast<ngraph::opset1::Constant>(C_init)->cast_vector<float>();

    auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hidden_size });
    auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hidden_size });
    H_t->set_friendly_name("hidden_state_1");
    C_t->set_friendly_name("cell_state_1");
    //Body
    auto X = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, 1, reshape1_shape[2] });
    auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hidden_size, reshape1_shape[2] }, {}, true);
    auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hidden_size, hidden_size }, {}, true);

    //lstm [1, 10], [1, 118], [1, 118] -> [1, 118], [1, 118]
    outFormShapes1 = { batch_size, reshape1_shape[2] };
    auto constantX = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, outFormShapes1);
    auto lstm1 = std::make_shared<ngraph::opset4::LSTMCell>(std::make_shared<ngraph::opset1::Reshape>(X, constantX, false),
        H_t, C_t,
        weightsNode, reccurrenceWeightsNode, hidden_size);

    auto H_o = lstm1->output(0);
    auto C_o = lstm1->output(1);

    //TensorIterator [1, 10, 49] [1, 118], [1, 118] -> [1, 118]
    auto body = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{ H_o, C_o }, ngraph::ParameterVector{ X, H_t, C_t });

    auto tensor_iterator = std::make_shared<ngraph::opset1::TensorIterator>();
    tensor_iterator->set_body(body);

    //input tensor shape: [1, 10, 49] chunk shape: [1, 1, 49]
    tensor_iterator->set_sliced_input(X, reshape1, 0, 1, 1, -1, 1);
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    auto out0 = tensor_iterator->get_iter_value(H_o, -1);

    const size_t output_size = 12;
    auto fc1 = ngraph::builder::makeFullyConnected(out0, ngPrc, output_size, true, { hidden_size, output_size }, { 1 }, { 1 });

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(fc1)};
    function = std::make_shared<ngraph::Function>(results, params, "Basic_LSTM_S");
}

void Basic_LSTM_S::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();
    Infer();

    const auto& actualOutputs = GetOutputs();
    auto referenceOutputs = CalculateRefs();

    Compare(referenceOutputs, actualOutputs);
}

std::vector<std::vector<std::uint8_t>> Basic_LSTM_S::CalculateRefs() {
    //For now TensorIterator is not implemented in ngraph interpreter so it is needed to validate with another reference
    auto reference_model = ngraph::clone_function(*function);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::UnrollTensorIterator>();
    manager.run_passes(reference_model);

    auto refCnnNetwork = InferenceEngine::CNNNetwork{ reference_model };
    auto refExecutableNetwork = core->LoadNetwork(refCnnNetwork, targetDevice);

    auto refInferRequest = refExecutableNetwork.CreateInferRequest();
    std::vector<InferenceEngine::InputInfo::Ptr> refInfos;
    for (const auto& input : refCnnNetwork.getInputsInfo()) {
        const auto& info = input.second;
        refInfos.push_back(info);
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        const auto& info = refInfos[i];

        refInferRequest.SetBlob(info->name(), input);
    }

    refInferRequest.Infer();

    auto refOutputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto& output : refCnnNetwork.getOutputsInfo()) {
        const auto& name = output.first;
        refOutputs.push_back(refInferRequest.GetBlob(name));
    }

    auto referenceOutputs = std::vector<std::vector<std::uint8_t>>(refOutputs.size());
    for (std::size_t i = 0; i < refOutputs.size(); ++i) {
        const auto& reference = refOutputs[i];
        const auto refSize = reference->byteSize();

        auto& expectedOutput = referenceOutputs[i];
        expectedOutput.resize(refSize);

        auto refMemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(reference);
        IE_ASSERT(refMemory);
        const auto refLockedMemory = refMemory->wmap();
        const auto referenceBuffer = refLockedMemory.as<const std::uint8_t*>();

        std::copy(referenceBuffer, referenceBuffer + refSize, expectedOutput.data());
    }

    return referenceOutputs;
}

TEST_P(Basic_LSTM_S, CompareWithRefImpl) {
    Run();
};

TEST_P(Basic_LSTM_S, CompareWithRefImpl_LowLatencyTransformation) {
    InferenceEngine::TensorDesc state_description(InferenceEngine::Precision::FP32,
                                                  InferenceEngine::SizeVector({1, hidden_size}),
                                                  InferenceEngine::Layout::NC);
    // Reshape
    auto params = ngraph::builder::makeParams(function->get_parameters().at(0)->get_element_type(), { {1, 49} });
    function->replace_parameter(0, params[0]);

    // todo: it is better to modify the model -> use ShapeOf() and Gather()
    std::vector<uint64_t> outFormShapes1 = { 1, 1, 49 };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{3}, outFormShapes1);
    auto param_target_inputs = function->get_parameters().at(0)->output(0).get_target_inputs();

    // replace hardcoded shape
    for (const auto& target : param_target_inputs.begin()->get_node()->input(1).get_source_output().get_target_inputs()) {
        target.replace_source_output(pattern1);
    }
    function->validate_nodes_and_infer_types();

    // Calculate References for the network before transformation passes
    auto referenceOutputs = CalculateRefs();

    // Apply LowLatency and UnrollTensorIterator transformations
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency>(); // LowLatency enables UnrollTI
    manager.run_passes(function);
    LoadNetwork();
    auto states = executableNetwork.QueryState();
    for (auto& state : states) {
        auto name = state.GetName();
        if (name.find("cell_state_1") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       cell_memory_init.data(), cell_memory_init.size());
            state.SetState(blob);
        } else if (name.find("hidden_state_1") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       hidden_memory_init.data(), hidden_memory_init.size());
            state.SetState(blob);
        } else {
            GTEST_FAIL() << "unknown memory state";
        }
    }
    // Run and compare
    Infer();
    const auto& actualOutputs = GetOutputs();
    Compare(referenceOutputs, actualOutputs);
};
}  // namespace LayerTestsDefinitions
