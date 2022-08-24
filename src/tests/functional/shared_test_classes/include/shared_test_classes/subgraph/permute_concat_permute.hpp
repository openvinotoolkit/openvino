// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {
typedef std::tuple<std::vector<std::vector<size_t>>,  // input shapes and permute shapes
                   InferenceEngine::Precision,        // Network precision
                   std::string                        // Device name
                   >
    PermuteConcatPermuteTuple;

class PermuteConcatPermute : public testing::WithParamInterface<PermuteConcatPermuteTuple>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PermuteConcatPermuteTuple>& obj);

protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& inputInfo) const override;

    int32_t range_{};
    int32_t start_{1};
    int32_t step_{1};
};
}  // namespace SubgraphTestsDefinitions
