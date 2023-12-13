// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using GridSampleParams = std::tuple<InferenceEngine::SizeVector,                                      // Data shape
                                    InferenceEngine::SizeVector,                                      // Grid shape
                                    decltype(ov::op::v9::GridSample::Attributes::align_corners),  // Align corners
                                    decltype(ov::op::v9::GridSample::Attributes::mode),           // Mode
                                    decltype(ov::op::v9::GridSample::Attributes::padding_mode),   // Padding mode
                                    InferenceEngine::Precision,                                       // Data precision
                                    InferenceEngine::Precision,                                       // Grid precision
                                    std::string>;                                                     // Device name

class GridSampleLayerTest : public testing::WithParamInterface<GridSampleParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GridSampleParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
