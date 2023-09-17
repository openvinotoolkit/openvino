// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include "common_test_utils/file_utils.hpp"
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
    return FileUtils::makePath<char>(
        FileUtils::makePath<char>(ov::test::utils::getExecutableDirectory(), TEST_MODELS), path);
}

void QuantizedModelsTests::runModel(const char* model, const LayerInputTypes& expected_layer_input_types, float thr) {
    threshold = thr;
    auto ie = getCore();
    auto network = ie->ReadNetwork(getModelFullPath(model));
    function = network.getFunction();
    Run();
    auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& name = node->get_friendly_name();
        if (expected_layer_input_types.count(name)) {
            ops_found++;
            const auto& expected_input_types = expected_layer_input_types.at(name);
            auto inputs = node->input_values();
            ASSERT_EQ(inputs.size(), expected_input_types.size());
            for (size_t i = 0; i < inputs.size(); i++)
                ASSERT_EQ(expected_input_types[i], inputs[i].get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}

TEST_P(QuantizedModelsTests, MaxPoolQDQ) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    runModel("max_pool_qdq.onnx", {{"890_original", {ngraph::element::u8}}}, 1e-5);
}

TEST_P(QuantizedModelsTests, MaxPoolFQ) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    runModel("max_pool_fq.onnx", {{"887_original", {ngraph::element::u8}}}, 1e-5);
}

TEST_P(QuantizedModelsTests, ConvolutionQDQ) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    // activations have type uint8 and weights int8
    runModel("convolution_qdq.onnx", {{"908_original", {ngraph::element::u8, ngraph::element::i8}}}, 1.5e-2);
}

TEST_P(QuantizedModelsTests, ConvolutionFQ) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    // activations have type uint8 and weights int8
    runModel("convolution_fq.onnx", {{"902_original", {ngraph::element::u8, ngraph::element::i8}}}, 1.5e-2);
}

} // namespace ONNXTestsDefinitions
