// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,               // Input shape
        size_t,                            // Split axis
        size_t,                            // Split number
        InferenceEngine::Precision,        // Network precision
        std::string,                       // Device name
        std::map<std::string, std::string> // Configuration
> ReluSplitReshapeTuple;

class ReluSplitReshape:
        public testing::WithParamInterface<ReluSplitReshapeTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReluSplitReshapeTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
