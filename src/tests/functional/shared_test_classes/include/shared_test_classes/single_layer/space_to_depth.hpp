// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

using spaceToDepthParamsTuple = typename std::tuple<
        std::vector<size_t>,                            // Input shape
        InferenceEngine::Precision,                     // Input precision
        ngraph::opset3::SpaceToDepth::SpaceToDepthMode, // Mode
        std::size_t,                                    // Block size
        std::string>;                                   // Device name>

class SpaceToDepthLayerTest : public testing::WithParamInterface<spaceToDepthParamsTuple>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<spaceToDepthParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
