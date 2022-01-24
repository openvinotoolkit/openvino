// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using clampParamsTuple = std::tuple<
    InferenceEngine::SizeVector,    // Input shape
    std::pair<float, float>,        // Interval [min, max]
    InferenceEngine::Precision,     // Net precision
    std::string>;                   // Device name

class ClampLayerTest : public testing::WithParamInterface<clampParamsTuple>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<clampParamsTuple>& obj);
protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
