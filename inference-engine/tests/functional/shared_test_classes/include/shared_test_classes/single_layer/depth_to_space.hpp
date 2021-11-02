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

using depthToSpaceSpecificParamsTuple = typename std::tuple<
        ngraph::opset3::DepthToSpace::DepthToSpaceMode, // Mode
        std::size_t>;                                   // Block size

using depthToSpaceParamsTuple = typename std::tuple<
        depthToSpaceSpecificParamsTuple,                // Specific params for DepthToSpace
        std::vector<size_t>,                            // Input shape
        InferenceEngine::Precision,                     // Input precision
        std::string>;                                   // Device name>

class DepthToSpaceLayerTest : public testing::WithParamInterface<depthToSpaceParamsTuple>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<depthToSpaceParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions