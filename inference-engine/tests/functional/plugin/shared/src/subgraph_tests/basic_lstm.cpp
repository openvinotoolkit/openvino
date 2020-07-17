// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

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
    threshold = 0.05f;

    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { {1, 490} });

    const size_t hidden_size = 118;
    const size_t batch_size = 1;

    outPrc = InferenceEngine::Precision::FP32;

    //Reshape_1 [1,490] -> [1, 10, 49]
    std::vector<uint64_t> outFormShapes1 = { batch_size, 10, 49 };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{3}, outFormShapes1);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

    auto reshape1_shape = reshape1->output(0).get_shape();
    auto H_init = ngraph::builder::makeConstant(ngPrc, { batch_size, hidden_size }, {}, true);
    auto C_init = ngraph::builder::makeConstant(ngPrc, { batch_size, hidden_size }, {}, true);

    auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hidden_size });
    auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, hidden_size });

    //Body
    auto X = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape{ batch_size, 1, reshape1_shape[2] });
    auto weightsNode = ngraph::builder::makeConstant(ngPrc, { 4 * hidden_size, reshape1_shape[2] }, {}, true);
    auto reccurrenceWeightsNode = ngraph::builder::makeConstant(ngPrc, { 4 * hidden_size, hidden_size }, {}, true);

    //lstm [1, 10], [1, 118], [1, 118] -> [1, 118], [1, 118]
    outFormShapes1 = { batch_size, reshape1_shape[2] };
    auto constantX = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, outFormShapes1);
    auto lstm1 = std::make_shared<ngraph::opset1::LSTMCell>(std::make_shared<ngraph::opset1::Reshape>(X, constantX, false),
        H_t, C_t,
        weightsNode, reccurrenceWeightsNode, hidden_size);

    auto H_o = lstm1->output(0);
    auto C_o = lstm1->output(1);

    //TensorIterator [1, 10, 49] [1, 118], [1, 118] -> [1, 118]
    auto body = std::make_shared<ngraph::opset1::TensorIterator::BodyLambda>(
        ngraph::OutputVector{ H_o, C_o }, ngraph::ParameterVector{ X, H_t, C_t });

    auto tensor_iterator = std::make_shared<ngraph::opset1::TensorIterator>();
    tensor_iterator->set_body(body);

    //input tensor shape: [1, 10, 49] chunk shape: [1, 1, 49]
    tensor_iterator->set_sliced_input(X, reshape1, 0, 1, 1, -1, 1);
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    auto out0 = tensor_iterator->get_iter_value(H_o, -1);

    //TODO: matmul not working on GNA
    //const size_t output_size = 12;
    //auto fc1 = ngraph::builder::makeFullyConnected(out0, ngPrc, output_size, true, {hidden_size, output_size});

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(out0)};
    function = std::make_shared<ngraph::Function>(results, params, "Basic_LSTM_S");
}

void Basic_LSTM_S::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ConfigurePlugin();
    LoadNetwork();
    Infer();

    const auto& actualOutputs = GetOutputs();

    //For now TensorIterator is not implemented in ngraph interpreter so it is needed to validate with another reference
    core = PluginCache::get().ie(CommonTestUtils::DEVICE_CPU);
    ConfigurePlugin();

    auto refCnnNetwork = InferenceEngine::CNNNetwork{ function };
    auto refExecutableNetwork = core->LoadNetwork(refCnnNetwork, CommonTestUtils::DEVICE_CPU);

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

        auto CPUmemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(reference);
        IE_ASSERT(CPUmemory);
        const auto CPUlockedMemory = CPUmemory->wmap();
        const auto referenceBuffer = CPUlockedMemory.as<const std::uint8_t*>();

        std::copy(referenceBuffer, referenceBuffer + refSize, expectedOutput.data());
    }

    Compare(referenceOutputs, actualOutputs);
}

TEST_P(Basic_LSTM_S, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
