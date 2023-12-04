// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        std::pair<size_t, size_t>,           // Third dimenstion and hidden size
        size_t,                              // Number of Cells
        bool,                                // Decompose LSTMCell
        std::pair<float, float>              // Input and weights range
> basicLstmParams;

class Basic_LSTM_S : public testing::WithParamInterface<basicLstmParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<basicLstmParams>& obj);

    void Run() override;
    static std::shared_ptr<ngraph::Function> GetNetwork(size_t thirdDimOut,
        size_t hiddenSize,
        size_t num_cells = 10,
        std::pair<float, float> weights_range = {0.f, 10.f},
        const InferenceEngine::Precision& netPrecission = InferenceEngine::Precision::FP32,
        std::vector<float>* hidden_memory_init_out = nullptr,
        std::vector<float>* cell_memory_init_out = nullptr);
    void GenerateInputs() override;
protected:
    void LoadNetwork() override;
    void Infer() override;

    size_t hidden_size;
    size_t third_dim;
    std::pair<float, float> weights_range;
    std::vector<float> hidden_memory_init;
    std::vector<float> cell_memory_init;
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
};

}  // namespace SubgraphTestsDefinitions
