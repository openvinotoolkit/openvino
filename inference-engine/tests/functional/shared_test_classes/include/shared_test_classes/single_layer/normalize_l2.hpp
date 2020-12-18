// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

using NormalizeL2LayerTestParams = std::tuple<
        std::vector<int64_t>,               // axes
        float,                              // eps
        ngraph::op::EpsMode,                // eps_mode
        InferenceEngine::SizeVector,        // inputShape
        InferenceEngine::Precision,         // netPrecision
        std::string                         // targetDevice
>;

class NormalizeL2LayerTest : public testing::WithParamInterface<NormalizeL2LayerTestParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2LayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

