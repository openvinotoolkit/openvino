// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
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
        uint32_t,                            // Input chunk size
        uint32_t,                            // Input chunk number
        uint32_t,                            // Const chunk size
        uint32_t                             // Const chunk number
> ConstStridedSliceConcatParams;

class ConstStridedSliceConcatTest : public testing::WithParamInterface<ConstStridedSliceConcatParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConstStridedSliceConcatParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
