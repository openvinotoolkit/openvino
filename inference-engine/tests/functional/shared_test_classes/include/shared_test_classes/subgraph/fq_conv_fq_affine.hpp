// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>,              // levels
        std::vector<float>                // input generator data: low, high, resolution
> FqSpecificParams;

typedef std::tuple<
        std::vector<size_t>,                 // Kernel Shape
        std::vector<size_t>,                 // Strides
        size_t,                              // Input channels
        size_t                               // Output channels
> ConvParams;

typedef std::tuple<
        FqSpecificParams,
        ConvParams,
        bool,                              // Permute after convolution
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::SizeVector,       // Input shapes
        LayerTestsUtils::TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional backend configuration and alis name to it
> FqConvFqAffineTestParamsSet;

class FqConvFqAffineTest : public testing::WithParamInterface<FqConvFqAffineTestParamsSet>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FqConvFqAffineTestParamsSet> obj);

protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    float inputDataMin        = 0.0;
    float inputDataMax        = 10.0;
    float inputDataResolution = 1.0;
    int32_t  seed = 1;
};

}  // namespace SubgraphTestsDefinitions
