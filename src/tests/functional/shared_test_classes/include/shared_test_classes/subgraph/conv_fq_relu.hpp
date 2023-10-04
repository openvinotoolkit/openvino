// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        size_t,                           // levels
        std::vector<float>,               // input generator data: low, high, resolution
        float                             // convolution weights' FQ min and max value
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
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::SizeVector,       // Input shapes
        LayerTestsUtils::TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional backend configuration and alis name to it
> ConvFqReluTestParamsSet;

class ConvFqReluTest : public testing::WithParamInterface<ConvFqReluTestParamsSet>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvFqReluTestParamsSet>& obj);

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
