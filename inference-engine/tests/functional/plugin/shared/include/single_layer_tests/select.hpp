// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <functional_test_utils/layer_test_utils.hpp>

#include "ngraph_functions/select.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,  // mask, then, else shapes
        InferenceEngine::Precision,        // then, else precision
        ngraph::op::AutoBroadcastSpec,     // broadcast
        std::string> selectTestParams;   // Device name

class SelectLayerTest : public LayerTestsUtils::LayerTestsCommonDeprecated<selectTestParams> {
public:
    NGraphFunctions::Select layer;
    std::vector<std::vector<size_t>> inputShapes;
    ngraph::op::AutoBroadcastSpec broadcast;

    static std::string getTestCaseName(const testing::TestParamInfo <selectTestParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions