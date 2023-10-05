// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        std::vector<size_t>,                 // Input Shapes
        size_t,                              // Num of Split outputs (concat inputs)
        bool                                 // with FC or not
> SplitConcatMultiInputsParams;


class SplitConcatMultiInputsTest : public testing::WithParamInterface<SplitConcatMultiInputsParams>,
                                   public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SplitConcatMultiInputsParams> obj);

protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

    float inputDataMin        = 0.0;
    float inputDataMax        = 0.2;
    float inputDataResolution = 1;
    int32_t  seed = 1;
};
}  // namespace SubgraphTestsDefinitions
