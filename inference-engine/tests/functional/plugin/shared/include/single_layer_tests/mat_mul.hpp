// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

struct ShapeRelatedParams {
    InferenceEngine::SizeVector firstInputShape, secondInputShape;
    bool transposeA, transposeB;
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

namespace LayerTestsDefinitions {

class MatMulTest : public testing::WithParamInterface<MatMulLayerTestParamsSet>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
