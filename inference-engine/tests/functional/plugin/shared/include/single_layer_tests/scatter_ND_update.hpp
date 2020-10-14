// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
using sliceSelcetInShape = std::tuple<
        std::vector<size_t>,    // input shape
        std::vector<size_t>,    // indices shape
        std::vector<size_t>,    // indices value
        std::vector<size_t>>;   // update shape

using scatterNDUpdateParamsTuple = typename std::tuple<
        sliceSelcetInShape,                // Input description
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // indices precision
        std::string>;                      // Device name

class ScatterNDUpdateLayerTest : public testing::WithParamInterface<scatterNDUpdateParamsTuple>,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterNDUpdateParamsTuple> &obj);
    static std::vector<sliceSelcetInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>>& inputShapes);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions