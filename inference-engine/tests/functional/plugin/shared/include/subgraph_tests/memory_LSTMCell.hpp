// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include <ie_core.hpp>

namespace SubgraphTestsDefinitions {
typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Input size
    size_t,                             // Hidden size
    std::map<std::string, std::string>  // Configuration
> memoryLSTMCellParams;

class MemoryLSTMCellTest : public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<memoryLSTMCellParams> {
private:
    // you have to Unroll TI manually and remove memory untill ngraph supports it
    void switchToNgraphFriendlyModel();
    // since we switching models we need to generate and save weights biases and inputs in SetUp
    std::vector<float> input_bias;
    std::vector<float> input_weights;
    std::vector<float> hidden_memory_init;
    std::vector<float> cell_memory_init;
    std::vector<float> weights_vals;
    std::vector<float> reccurrenceWeights_vals;
    std::vector<float> bias_vals;
protected:
    void SetUp() override;
    void Run() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<memoryLSTMCellParams> &obj);
};
} // namespace SubgraphTestsDefinitions
