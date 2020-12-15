// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

typedef std::tuple<
    InferenceEngine::Precision,          // Network Precision
    std::string,                         // Target Device
    std::map<std::string, std::string>   // Configuration
> basicLstmParams;

namespace LayerTestsDefinitions {

class Basic_LSTM_S : public testing::WithParamInterface<basicLstmParams>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<basicLstmParams> obj);

    void Run() override;
    static std::shared_ptr<ngraph::Function> GetNetwork(size_t thirdDimOut,
        size_t hiddenSize,
        const InferenceEngine::Precision& netPrecission = InferenceEngine::Precision::FP32,
        std::vector<float>* hidden_memory_init_out = nullptr,
        std::vector<float>* cell_memory_init_out = nullptr);
protected:
    size_t hidden_size;
    std::vector<float> hidden_memory_init;
    std::vector<float> cell_memory_init;
    void SetUp() override;
    std::vector<std::vector<std::uint8_t>> CalculateRefs() override;
};

}  // namespace LayerTestsDefinitions
