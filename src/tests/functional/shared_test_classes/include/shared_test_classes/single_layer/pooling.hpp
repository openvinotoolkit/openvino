// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ngraph::helpers::PoolingTypes,  // Pooling type, max or avg
        std::vector<size_t>,            // Kernel size
        std::vector<size_t>,            // Stride
        std::vector<size_t>,            // Pad begin
        std::vector<size_t>,            // Pad end
        ngraph::op::RoundingType,       // Rounding type
        ngraph::op::PadType,            // Pad type
        bool                            // Exclude pad
> poolSpecificParams;
typedef std::tuple<
        poolSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        std::vector<size_t>,            // Input shape
        std::string                     // Device name
> poolLayerTestParamsSet;

typedef std::tuple<
        poolSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        size_t,                         // Channel number
        std::string                     // Device name
> globalPoolLayerTestParamsSet;

typedef std::tuple<
        std::vector<size_t>,            // Kernel size
        std::vector<size_t>,            // Stride
        std::vector<size_t>,            // Dilation
        std::vector<size_t>,            // Pad begin
        std::vector<size_t>,            // Pad end
        ngraph::element::Type_t,        // Index element type
        int64_t,                        // Axis
        ngraph::op::RoundingType,       // Rounding type
        ngraph::op::PadType             // Pad type
> maxPoolV8SpecificParams;

typedef std::tuple<
        maxPoolV8SpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        std::vector<size_t>,            // Input shape
        std::string                     // Device name
> maxPoolV8LayerTestParamsSet;

class PoolingLayerTest : public testing::WithParamInterface<poolLayerTestParamsSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

class GlobalPoolingLayerTest : public testing::WithParamInterface<globalPoolLayerTestParamsSet>,
                               virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<globalPoolLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

class MaxPoolingV8LayerTest : public testing::WithParamInterface<maxPoolV8LayerTestParamsSet>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<maxPoolV8LayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
