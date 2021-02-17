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
struct ShapeRelatedParams {
    std::pair<InferenceEngine::SizeVector, bool> input1, input2;
};

typedef std::tuple<
        ShapeRelatedParams,
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        ngraph::helpers::InputLayerType,   // Secondary input type
        LayerTestsUtils::TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

class MatMulTest : public testing::WithParamInterface<MatMulLayerTestParamsSet>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj);
    static std::vector<ShapeRelatedParams> combineShapes(const std::vector<std::vector<size_t>>& firstInputShapes,
                                                         const std::vector<std::vector<size_t>>& secondInputShapes,
                                                         bool transposeA,
                                                         bool transposeB);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
