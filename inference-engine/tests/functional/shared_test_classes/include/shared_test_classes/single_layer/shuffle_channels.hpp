// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        int, // axis
        int  // group
> ShuffleChannelsSpecificParams;

typedef std::tuple<
        ShuffleChannelsSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shapes
        LayerTestsUtils::TargetDevice   // Device name
> ShuffleChannelsLayerTestParams;

class ShuffleChannelsLayerTest : public testing::WithParamInterface<ShuffleChannelsLayerTestParams>,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsLayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
