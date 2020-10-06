// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include <ie_core.hpp>

namespace SubgraphTestsDefinitions {
enum class midOutputType {
    Sum,
    Sub,
    Mul,
};

typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Input size
    midOutputType,                      // Type of layer that will be an output
    std::map<std::string, std::string>  // Configuration
> outputBeforeActivationParams;

std::ostream& operator<< (std::ostream& os, const midOutputType& oType);

class OutputBeforeActivation : public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<outputBeforeActivationParams> {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<outputBeforeActivationParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
};
} // namespace SubgraphTestsDefinitions
