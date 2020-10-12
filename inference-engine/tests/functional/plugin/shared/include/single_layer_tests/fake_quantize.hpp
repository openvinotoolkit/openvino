// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

typedef std::tuple<
        size_t,              // levels
        std::vector<size_t>, // const inputs shape
        std::vector<float>,  // fake quantize inputLow, inputHigh, outputLow, outputHigh or empty for random
        std::vector<float>   // input generator data: low, high, resolution
> fqSpecificParams;
typedef std::tuple<
        fqSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::SizeVector,       // Input shapes
        LayerTestsUtils::TargetDevice,     // Device name

        std::pair<std::string, std::map<std::string, std::string>> // Additional backend configuration and alis name to it
> fqLayerTestParamsSet;
namespace LayerTestsDefinitions {


class FakeQuantizeLayerTest : public testing::WithParamInterface<fqLayerTestParamsSet>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqLayerTestParamsSet> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
protected:
    void SetUp() override;
    void UpdateSeed();

 protected:
    float inputDataMin        = 0.0;
    float inputDataMax        = 10.0;
    float inputDataResolution = 1.0;
    int32_t  seed = 1;
};

}  // namespace LayerTestsDefinitions
