// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using axisShapeInShape = std::tuple<
        std::vector<size_t>,    // input shape
        std::vector<size_t>,    // update shape
        int>;                   // axis

using scatterElementsUpdateParamsTuple = typename std::tuple<
        axisShapeInShape,                  // shape description
        std::vector<size_t>,               // indices value
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // indices precision
        std::string>;                      // Device name

using scatterElementsUpdate12ParamsTuple = typename std::tuple<
        axisShapeInShape,                  // shape description
        std::vector<int64_t>,              // indices value
        ov::op::v12::ScatterElementsUpdate::Reduction,  // Reduce mode
        bool,                              // Use init value
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // indices precision
        std::string>;                      // Device name

class ScatterElementsUpdateLayerTest : public testing::WithParamInterface<scatterElementsUpdateParamsTuple>,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterElementsUpdateParamsTuple> &obj);
    static std::vector<axisShapeInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& inputShapes);

protected:
    void SetUp() override;
};

class ScatterElementsUpdate12LayerTest : public testing::WithParamInterface<scatterElementsUpdate12ParamsTuple>,
                                         virtual public LayerTestsUtils::LayerTestsCommon  {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterElementsUpdate12ParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
