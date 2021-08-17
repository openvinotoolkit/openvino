// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string Basic_LSTM_S::getTestCaseName(testing::TestParamInfo<basicLstmParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::pair<size_t, size_t> size_params;
    bool decompose;
    std::tie(netPrecision, targetDevice, configuration, size_params, decompose) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    result << "_TD=" << size_params.first;
    result << "_HS=" << size_params.second;
    result << "_D=" << decompose;
    return result.str();
}

void Basic_LSTM_S::SetUp() {
    threshold = 0.1f;

    InferenceEngine::Precision netPrecision;
    std::pair<size_t, size_t> size_params;
    bool decompose;
    std::tie(netPrecision, targetDevice, configuration, size_params, decompose) = this->GetParam();
    third_dim = size_params.first;
    hidden_size = size_params.second;
    outPrc = InferenceEngine::Precision::FP32;

    function = GetNetwork(size_params.first, size_params.second, netPrecision, &hidden_memory_init, &cell_memory_init);

    if (decompose) {
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
        manager.run_passes(function);
    }
}

std::shared_ptr<ngraph::Function> Basic_LSTM_S::GetNetwork(size_t thirdDimOut,
                                                           size_t hiddenSize,
                                                           const InferenceEngine::Precision& netPrecission,
                                                           std::vector<float>* hidden_memory_init_out,
                                                           std::vector<float>* cell_memory_init_out) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecission);

    auto params = ngraph::builder::makeParams(ngPrc, { {1, 10 * thirdDimOut} });

    const size_t batch_size = 1;

    //Reshape_1 [1,thirdDimOut*10] -> [1, 10, thirdDimOut]
    std::vector<uint64_t> outFormShapes1 = { batch_size, 10, thirdDimOut };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 3 }, outFormShapes1);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

    auto reshape1_shape = reshape1->output(0).get_shape();
    auto H_init = ngraph::builder::makeConstant<float>(ngPrc, { batch_size, hiddenSize }, {}, true);
    auto C_init = ngraph::builder::makeConstant<float>(ngPrc, { batch_size, hiddenSize }, {}, true);
    if (hidden_memory_init_out != nullptr) {
        *hidden_memory_init_out = std::static_pointer_cast<ngraph::opset1::Constant>(H_init)->cast_vector<float>();
    }
    if (cell_memory_init_out != nullptr) {
        *cell_memory_init_out = std::static_pointer_cast<ngraph::opset1::Constant>(C_init)->cast_vector<float>();
    }
    auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hiddenSize });
    auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hiddenSize });
    H_t->set_friendly_name("hidden_state_1");
    C_t->set_friendly_name("cell_state_1");
    //Body
    auto X = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, 1, reshape1_shape[2] });
    auto weightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, reshape1_shape[2] }, {}, true);
    auto reccurrenceWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, { 4 * hiddenSize, hiddenSize }, {}, true);

    //lstm [1, 10], [1, 118], [1, 118] -> [1, 118], [1, 118]
    outFormShapes1 = { batch_size, reshape1_shape[2] };
    auto constantX = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ 2 }, outFormShapes1);
    auto lstm1 = std::make_shared<ngraph::opset4::LSTMCell>(std::make_shared<ngraph::opset1::Reshape>(X, constantX, false),
        H_t, C_t,
        weightsNode, reccurrenceWeightsNode, hiddenSize);

    auto H_o = lstm1->output(0);
    auto C_o = lstm1->output(1);

    //TensorIterator [1, 10, thirdDimOut] [1, 118], [1, 118] -> [1, 118]
    auto body = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{ H_o, C_o }, ngraph::ParameterVector{ X, H_t, C_t });

    auto tensor_iterator = std::make_shared<ngraph::opset1::TensorIterator>();
    tensor_iterator->set_body(body);

    //input tensor shape: [1, 10, thirdDimOut] chunk shape: [1, 1, thirdDimOut]
    tensor_iterator->set_sliced_input(X, reshape1, 0, 1, 1, -1, 1);
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    auto out0 = tensor_iterator->get_iter_value(H_o, -1);

    const size_t output_size = 12;
    auto fc1 = ngraph::builder::makeFullyConnected(out0, ngPrc, output_size, true, { hiddenSize, output_size }, { 1 }, { 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fc1) };
    return std::make_shared<ngraph::Function>(results, params, "Basic_LSTM_S");
}

void Basic_LSTM_S::GenerateInputs() {
    // Generate inputs can be called before actual network loading in case lf LowLatencyTransformation
    auto inputs_function = ngraph::clone_function(*function);
    auto inputsCnnNetwork = InferenceEngine::CNNNetwork{ inputs_function };
    auto inputsExecutableNetwork = core->LoadNetwork(inputsCnnNetwork, targetDevice);
    const auto& inputsInfo = inputsExecutableNetwork.GetInputsInfo();
    const auto& functionParams = inputs_function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;
        auto blob = GenerateInput(*info);
        inputs.push_back(blob);
    }
}

void Basic_LSTM_S::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();
    GenerateInputs();
    Infer();

    const auto& actualOutputs = GetOutputs();
    auto referenceOutputs = CalculateRefs();

    Compare(referenceOutputs, actualOutputs);
}

std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> Basic_LSTM_S::CalculateRefs() {
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

    auto referenceOutputs = std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>(refOutputs.size());
    for (std::size_t i = 0; i < refOutputs.size(); ++i) {
        const auto& reference = refOutputs[i];
        const auto refSize = reference->byteSize();

        referenceOutputs[i].first = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(reference->getTensorDesc().getPrecision());
        auto& expectedOutput = referenceOutputs[i].second;
        expectedOutput.resize(refSize);

        auto refMemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(reference);
        IE_ASSERT(refMemory);
        const auto refLockedMemory = refMemory->wmap();
        const auto referenceBuffer = refLockedMemory.as<const std::uint8_t*>();

        std::copy(referenceBuffer, referenceBuffer + refSize, expectedOutput.data());
    }

    return referenceOutputs;
}

}  // namespace SubgraphTestsDefinitions
