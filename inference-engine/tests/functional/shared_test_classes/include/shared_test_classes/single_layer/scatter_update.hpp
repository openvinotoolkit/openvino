// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using axisUpdateShapeInShape = std::tuple<
        std::vector<size_t>,    // input shape
        std::vector<size_t>,    // indices shape
        std::vector<size_t>,    // update shape
        int>;                   // axis

using scatterUpdateParamsTuple = typename std::tuple<
        axisUpdateShapeInShape,                  // shape description
        std::vector<size_t>,               // indices value
        InferenceEngine::Precision,        // input precision
        InferenceEngine::Precision,        // indices precision
        std::string>;                      // Device name

class ScatterUpdateLayerTest : public testing::WithParamInterface<scatterUpdateParamsTuple>,
                               virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterUpdateParamsTuple> &obj);
    static std::vector<axisUpdateShapeInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& inputShapes);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
