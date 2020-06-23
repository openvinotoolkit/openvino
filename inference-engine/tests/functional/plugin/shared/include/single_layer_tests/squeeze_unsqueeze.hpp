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
using ShapeAxesTuple = std::pair<std::vector<size_t>, std::vector<int>>;

typedef std::tuple<
        ShapeAxesTuple,                 // InputShape, Squeeze indexes
        ngraph::helpers::SqueezeOpType, // OpType
        InferenceEngine::Precision,     // Net precision
        std::string                     // Target device name
> squeezeParams;

class SqueezeUnsqueezeLayerTest : public testing::WithParamInterface<squeezeParams>,
                       public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<squeezeParams> obj);
    static std::vector<ShapeAxesTuple> combineShapes(const std::map<std::vector<size_t>, std::vector<std::vector<int>>>& inputShapes);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions