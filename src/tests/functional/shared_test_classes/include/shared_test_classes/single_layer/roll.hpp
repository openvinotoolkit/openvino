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
        std::vector<int64_t>,        // Shift
        std::vector<int64_t>,        // Axes
        std::string> rollParams;   // Device name

class RollLayerTest : public testing::WithParamInterface<rollParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<rollParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
