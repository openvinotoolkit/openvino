// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::SizeVector, // Input shapes
        InferenceEngine::Precision,  // Input precision
        int64_t,                     // Axis
        bool,                        // Exclusive
        bool,                        // Reverse
        std::string> cumSumParams;   // Device name

class CumSumLayerTest : public testing::WithParamInterface<cumSumParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<cumSumParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
