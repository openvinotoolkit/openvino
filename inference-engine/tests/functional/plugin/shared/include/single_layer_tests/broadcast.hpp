// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"

typedef std::tuple<
        ngraph::op::BroadcastModeSpec, // mode
        std::vector<size_t>,            // target shape
        std::vector<size_t>            // axes mapping
> broadcastSpecificParams;
typedef std::tuple<
        broadcastSpecificParams,
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Device name
> broadcastLayerTestParamsSet;

namespace LayerTestsDefinitions {

class BroadcastLayerTest : public testing::WithParamInterface<broadcastLayerTestParamsSet>,
                                 public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<broadcastLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
