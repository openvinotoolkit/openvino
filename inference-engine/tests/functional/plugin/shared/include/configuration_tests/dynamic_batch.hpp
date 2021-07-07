// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <tuple>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ConfigurationTestsDefinitions {
typedef std::tuple<
    std::string,                       // Device
    InferenceEngine::Precision,        // Network precision
    std::vector<size_t>,               // Batch sizes
    bool,                              // Asynchronous execution
    std::map<std::string, std::string> // Additional configuration
> dynamicBatchTestParams;

class DynamicBatchTest : public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<dynamicBatchTestParams> {
private:
    bool run_async = false;
    size_t max_batch_size = 0;
    std::vector<size_t> batch_sizes;
    std::vector<std::vector<InferenceEngine::Blob::Ptr>> reference_inputs;
    std::vector<std::vector<InferenceEngine::Blob::Ptr>> scaled_inputs;
    std::vector<std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>> reference_outputs;
    std::vector<std::vector<InferenceEngine::Blob::Ptr>> actual_outputs;
    std::vector<InferenceEngine::InferRequest> infer_requests;
protected:
    void SetUp() override;
    void Run() override;

    void LoadNetwork() override;
    void Infer() override;
    void Validate() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<dynamicBatchTestParams> &obj);
};
} // namespace ConfigurationTestsDefinitions
