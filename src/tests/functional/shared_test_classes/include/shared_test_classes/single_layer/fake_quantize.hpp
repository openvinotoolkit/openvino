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

// seed selected using current cloc time
#define USE_CLOCK_TIME 1
// seed started from default value, and incremented every time using big number like 9999
#define USE_INCREMENTAL_SEED 2

/**
 * redefine this seed to reproduce issue with given seed that can be read from gtest logs
 */
#define BASE_SEED   123
#define NGRAPH_SEED 123

namespace LayerTestsDefinitions {


typedef std::tuple<
        size_t,                         // fake quantize levels
        std::vector<size_t>,            // fake quantize inputs shape
        std::vector<float>,             // fake quantize (inputLow, inputHigh, outputLow, outputHigh) or empty for random
        std::vector<float>,             // input generator data (low, high, resolution) or empty for default
        ngraph::op::AutoBroadcastSpec   // fake quantize broadcast mode
> fqSpecificParams;
typedef std::tuple<
        fqSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        InferenceEngine::SizeVector,       // Input shapes
        LayerTestsUtils::TargetDevice,     // Device name

        std::pair<std::string, std::map<std::string, std::string>> // Additional backend configuration and alis name to it
> fqLayerTestParamsSet;

class FakeQuantizeLayerTest : public testing::WithParamInterface<fqLayerTestParamsSet>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<fqLayerTestParamsSet>& obj);
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
