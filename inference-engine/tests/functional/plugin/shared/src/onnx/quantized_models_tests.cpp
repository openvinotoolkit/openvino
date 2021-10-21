// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

void QuantizedModelsTests::runModel(const char* model, const std::unordered_map<std::string, ngraph::element::Type_t>& expected_layer_types) {
    auto ie = getCore();
    auto network = ie->ReadNetwork(getModelFullPath(model));
    function = network.getFunction();
    Run();
    auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
    int ops_found = 0;
    for (const auto& node : runtime_function->get_ordered_ops()) {
        const auto& name = node->get_friendly_name();
        if (expected_layer_types.count(name)) {
            ops_found++;
            ASSERT_EQ(expected_layer_types.at(name), node->get_element_type());
        }
    }
    ASSERT_GT(ops_found, 0);
}

TEST_P(QuantizedModelsTests, MaxPoolQDQ) {
    runModel("max_pool_qdq.onnx", {{"890_original", ngraph::element::u8}});
}

TEST_P(QuantizedModelsTests, MaxPoolFQ) {
    runModel("max_pool_fq.onnx", {{"887_original", ngraph::element::u8}});
}
} // namespace ONNXTestsDefinitions
