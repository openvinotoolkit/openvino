// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_tests/include/infer_consistency_test.hpp"
#include "ngraph_functions/builders.hpp"

std::shared_ptr<ngraph::Function> getDefaultGraph() {
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 224, 224});
    size_t spatialDims = 2;
    std::vector<ptrdiff_t> padBegin(spatialDims, 0), padEnd(spatialDims, 0);
    ngraph::Shape strides(spatialDims, 1);
    auto weights = ngraph::builder::makeConstant<float>(ov::element::f32, {64, 3, 7, 7}, {}, true);
    auto conv1 = std::make_shared<ov::opset8::Convolution>(input, weights, strides, padBegin, padEnd, strides);
    auto gamma = ngraph::builder::makeConstant<float>(ov::element::f32, {64}, {}, true);
    auto beta = ngraph::builder::makeConstant<float>(ov::element::f32, {64}, {}, true);
    auto mean = ngraph::builder::makeConstant<float>(ov::element::f32, {64}, {}, true);
    auto variance = ngraph::builder::makeConstant<float>(ov::element::f32, {64}, {}, true);
    auto batchNorm1 = std::make_shared<ov::opset8::BatchNormInference>(conv1, gamma, beta, mean, variance, 1e-5);
    auto relu1 = std::make_shared<ov::opset8::Relu>(batchNorm1);
    auto pool = std::make_shared<ov::opset8::AvgPool>(relu1, strides, ov::Shape{1, 1},
                                                      ov::Shape{1, 1}, ov::Shape{4, 4}, true);
    return std::make_shared<ngraph::Function>(ngraph::OutputVector{pool}, ngraph::ParameterVector{input},
                                              "autoSampleGraph");
}


namespace SubgraphTestsDefinitions {
void AutoInferConsistency::SetUp() {
    std::string modelPath;
    std::tie(_inferCount, modelPath, targetDevice, _baseDevice,
             configuration, _baseConfig) = WithParamInterface::GetParam();
    if (modelPath.empty()) {
        function = getDefaultGraph();
    } else {
        cnnNetwork = core->ReadNetwork(modelPath);
        function = cnnNetwork.getFunction();
    }
}

std::string AutoInferConsistency::getTestCaseName(const testing::TestParamInfo<ParamType> &obj) {
    size_t inferCount = 0;
    std::string modelPath;
    std::string targetDevice;
    std::string baseDevice;
    std::map<std::string, std::string>  autoConfigure;
    std::map<std::string, std::string>  baseConfigure;
    std::tie(inferCount, modelPath, targetDevice,
             baseDevice, autoConfigure, baseConfigure) = obj.param;
    std::ostringstream result;
    std::string ovHint("");
    if (autoConfigure.find(CONFIG_KEY(PERFORMANCE_HINT)) != autoConfigure.end())
        ovHint = autoConfigure.at(CONFIG_KEY(PERFORMANCE_HINT));

    result << "InferCount=" << inferCount << "_";
    result << "Dev=" << targetDevice << "_";
    result << "BaseDev=" << baseDevice << "_";
    result << "Hint=" << ovHint;
    return result.str();
}

std::vector<InferenceEngine::Blob::Ptr> AutoInferConsistency::getBaseNetOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : _baseExecNet.GetOutputsInfo()) {
        const auto &name = output.first;
        outputs.push_back(_baseInferRequest.GetBlob(name));
    }
    return outputs;
}

void AutoInferConsistency::loadBaseNet() {
    _baseExecNet = core->LoadNetwork(cnnNetwork, _baseDevice, _baseConfig);
}

void AutoInferConsistency::inferBaseNet() {
    _baseInferRequest = _baseExecNet.CreateInferRequest();

    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;
        auto blob = inputs[i];
        _baseInferRequest.SetBlob(info->name(), blob);
    }
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        _baseInferRequest.SetBatch(batchSize);
    }
    _baseInferRequest.Infer();
}

TEST_P(AutoInferConsistency, CompareWithDirectHW) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    //Load function to auto device
    LoadNetwork();
    //Load function to base device
        loadBaseNet();
    for (size_t i = 0; i < getInferCount(); i++) {
        GenerateInputs();
        //Infer in AUTO
        Infer();
        //Infer in Base HW
        inferBaseNet();
        const auto &actualOutputs = GetOutputs();
        const auto &expectedOutputs = getBaseNetOutputs();
        for (size_t j = 0; j < actualOutputs.size(); j++) {
            Compare(expectedOutputs[j], actualOutputs[j]);
        }
    }
}
namespace AutoInfer100Throughput {
    std::map<std::string, std::string>  baseDeviceConfig = {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(THROUGHPUT)}};
    std::map<std::string, std::string>  autoDeviceConfig = {baseDeviceConfig.begin(), baseDeviceConfig.end()};

    const auto param100InferThroughputCPU = ::testing::Combine(
                ::testing::Values(100),
                ::testing::Values(std::string()),
                ::testing::Values("AUTO:CPU"),
                ::testing::Values("CPU"),
                ::testing::Values(autoDeviceConfig),
                ::testing::Values(baseDeviceConfig));
    const auto param100InferThroughputGPU = ::testing::Combine(
            ::testing::Values(100),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    INSTANTIATE_TEST_SUITE_P(AutoCPUTest, AutoInferConsistency, param100InferThroughputCPU, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoGPUTest, AutoInferConsistency, param100InferThroughputGPU, AutoInferConsistency::getTestCaseName);
} // namespace AutoInfer100Throughput

namespace AutoInfer100Latency {
    std::map<std::string, std::string>  baseDeviceConfig = {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}};
    std::map<std::string, std::string>  autoDeviceConfig = {baseDeviceConfig.begin(), baseDeviceConfig.end()};
    const auto param100InferLatencyCPU = ::testing::Combine(
            ::testing::Values(100),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:CPU"),
            ::testing::Values("CPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    const auto param100InferLatencyGPU = ::testing::Combine(
            ::testing::Values(100),
            ::testing::Values(std::string()),
            ::testing::Values("AUTO:GPU"),
            ::testing::Values("GPU"),
            ::testing::Values(autoDeviceConfig),
            ::testing::Values(baseDeviceConfig));
    INSTANTIATE_TEST_SUITE_P(AutoCPUTest, AutoInferConsistency, param100InferLatencyCPU, AutoInferConsistency::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(AutoGPUTest, AutoInferConsistency, param100InferLatencyGPU, AutoInferConsistency::getTestCaseName);
} // namespace AutoInfer100Latency
} // namespace SubgraphTestsDefinitions

