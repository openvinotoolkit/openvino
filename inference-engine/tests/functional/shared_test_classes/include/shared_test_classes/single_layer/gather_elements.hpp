// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using Config = std::map<std::string, std::string>;

typedef std::tuple<
        std::vector<size_t>,               // Data shapes
        int,                               // axis
        int                                // indices axis dim
> GatherElementsParamsSubset;

typedef std::tuple<
        GatherElementsParamsSubset,
        InferenceEngine::Precision,        // Data precision
        InferenceEngine::Precision,        // Indices precision
        LayerTestsUtils::TargetDevice,     // Device name
        Config                             // Plugin config
> GatherElementsParams;

class GatherElementsLayerTest : public testing::WithParamInterface<GatherElementsParams>,
                                public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
