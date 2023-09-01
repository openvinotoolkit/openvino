// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::vector<int64_t> TileSpecificParams;
typedef std::tuple<
        TileSpecificParams,
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Device name
> TileLayerTestParamsSet;

class TileLayerTest : public testing::WithParamInterface<TileLayerTestParamsSet>,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TileLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
