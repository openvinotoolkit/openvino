// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
using SqueezeShape = std::pair<std::vector<size_t>, std::vector<int>>;
using InputShape = std::vector<size_t>;
using SqueezeAxes = std::vector<int>;

typedef std::tuple<
        SqueezeShape,                   // InputShape, Axes vector
        InferenceEngine::Precision,     // Net precision
        std::string,                    // Target device name
        bool                            // IsScalar
> squeezeParams;

class SqueezeLayerTest : public testing::WithParamInterface<squeezeParams>,
                       public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<squeezeParams> obj);
    static std::vector<SqueezeShape> combineShapes(const std::map<InputShape, std::vector<SqueezeAxes>>& inputShapes);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions