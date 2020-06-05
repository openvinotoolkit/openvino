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
        int, // axis
        int  // group
> shuffleChannelsSpecificParams;
typedef std::tuple<
        shuffleChannelsSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::SizeVector,    // Input shapes
        LayerTestsUtils::TargetDevice   // Device name
> shuffleChannelsLayerTestParamsSet;
namespace LayerTestsDefinitions {


class ShuffleChannelsLayerTest : public testing::WithParamInterface<shuffleChannelsLayerTestParamsSet>,
                                 public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<shuffleChannelsLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
