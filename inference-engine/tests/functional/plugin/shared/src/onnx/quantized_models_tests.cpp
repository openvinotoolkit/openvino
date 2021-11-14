// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exec_graph_info.hpp>
#include <file_utils.h>
#include "onnx/quantized_models_tests.hpp"

namespace ONNXTestsDefinitions {

std::string QuantizedModelsTests::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    std::string targetDevice = obj.param;
    std::ostringstream result;
    result << "device=" << targetDevice;
    return result.str();
}

void QuantizedModelsTests::SetUp() {
    targetDevice = this->GetParam();
}

static std::string getModelFullPath(const char* path) {
    return FileUtils::makePath<char>(TEST_MODELS, path);
}

void QuantizedModelsTests::runModel(const char* model, const LayerInputTypes& expected_layer_input_types, float thr) {
    threshold = thr;
    auto ie = getCore();
    auto network = ie->ReadNetwork(getModelFullPath(model));
    function = network.getFunction();
    Run();
    auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
    auto get_layer_type = [] (const std::shared_ptr<ngraph::Node>& node) -> const std::string& {
        const auto& rt_info = node->get_rt_info();
        auto it = rt_info.find(ExecGraphInfoSerialization::LAYER_TYPE);
        IE_ASSERT(it != rt_info.end());
        auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
        IE_ASSERT(value != nullptr);
        return value->get();
    };
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        auto it = expected_layer_input_types.find(get_layer_type(node));
        if (it != expected_layer_input_types.end()) {
            ops_found++;
            const auto& expected_input_types = it->second;
            auto inputs = node->input_values();
            ASSERT_EQ(inputs.size(), expected_input_types.size());
            for (size_t i = 0; i < inputs.size(); i++)
                ASSERT_EQ(expected_input_types[i], inputs[i].get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}

TEST_P(QuantizedModelsTests, MaxPoolQDQ) {
    runModel("max_pool_qdq.onnx", {{"Pooling", {ngraph::element::u8}}}, 1e-5);
}

TEST_P(QuantizedModelsTests, MaxPoolFQ) {
    runModel("max_pool_fq.onnx", {{"Pooling", {ngraph::element::u8}}}, 1e-5);
}

TEST_P(QuantizedModelsTests, ConvolutionQDQ) {
    // activations have type uint8 and weights int8
    runModel("convolution_qdq.onnx", {{"Convolution", {ngraph::element::u8, ngraph::element::i8}}}, 1.5e-2);
}

TEST_P(QuantizedModelsTests, ConvolutionFQ) {
    // activations have type uint8 and weights int8
    runModel("convolution_fq.onnx", {{"Convolution", {ngraph::element::u8, ngraph::element::i8}}}, 1.5e-2);
}

} // namespace ONNXTestsDefinitions
