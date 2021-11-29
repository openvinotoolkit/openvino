// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

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