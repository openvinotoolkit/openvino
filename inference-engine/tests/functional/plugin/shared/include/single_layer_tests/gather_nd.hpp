// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"


typedef std::tuple<
        std::vector<size_t>,               // Data shapes
        std::vector<size_t>,               // Indices shape
        int                                // batch dims
> GatherNDParamsSubset;

typedef std::tuple<
        GatherNDParamsSubset,
        InferenceEngine::Precision,        // Data precision
        InferenceEngine::Precision,        // Indices precision
        LayerTestsUtils::TargetDevice      // Device name
> GatherNDParams;

namespace LayerTestsDefinitions {

class GatherNDLayerTest : public testing::WithParamInterface<GatherNDParams>,
                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
