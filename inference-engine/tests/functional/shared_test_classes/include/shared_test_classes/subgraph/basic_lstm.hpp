// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        std::pair<size_t, size_t>,           // Third dimenstion and hidden size
        bool                                 // Decompose LSTMCell
> basicLstmParams;

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
    void GenerateInputs() override;
protected:
    size_t hidden_size;
    size_t third_dim;
    std::vector<float> hidden_memory_init;
    std::vector<float> cell_memory_init;
    void SetUp() override;
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override;
};

}  // namespace SubgraphTestsDefinitions
