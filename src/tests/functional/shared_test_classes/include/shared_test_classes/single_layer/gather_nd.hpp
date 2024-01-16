// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {
using Config = std::map<std::string, std::string>;

typedef std::tuple<
        std::vector<size_t>,               // Data shapes
        std::vector<size_t>,               // Indices shape
        int                                // batch dims
> GatherNDParamsSubset;

typedef std::tuple<
        GatherNDParamsSubset,
        InferenceEngine::Precision,        // Data precision
        InferenceEngine::Precision,        // Indices precision
        LayerTestsUtils::TargetDevice,     // Device name
        Config                             // Plugin config
> GatherNDParams;

class GatherNDLayerTest : public testing::WithParamInterface<GatherNDParams>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams> &obj);

protected:
    void SetUp() override;
};

class GatherND8LayerTest : public testing::WithParamInterface<GatherNDParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
