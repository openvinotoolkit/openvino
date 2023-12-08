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
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::vector<size_t>,                // Input shape
    std::map<std::string, std::string>  // Configuration
> TransposeAddParams;

class TransposeAdd : public testing::WithParamInterface<TransposeAddParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeAddParams> obj);

protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
