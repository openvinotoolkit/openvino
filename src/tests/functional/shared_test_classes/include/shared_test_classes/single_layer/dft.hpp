// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<InferenceEngine::SizeVector,  // Input shapes
                   InferenceEngine::Precision,   // Input precision
                   std::vector<int64_t>,         // Axes
                   std::vector<int64_t>,         // Signal size
                   ngraph::helpers::DFTOpType,
                   std::string>
    DFTParams;  // Device name

class DFTLayerTest : public testing::WithParamInterface<DFTParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DFTParams>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<InferenceEngine::SizeVector,  // Input shapes
                   InferenceEngine::Precision,   // Input precision
                   std::vector<int64_t>,         // Axes
                   std::vector<int64_t>,         // Signal size
                   ngraph::helpers::DFTOpType,
                   ngraph::helpers::DFTOpMode,
                   std::string>
    DFT9Params;  // Device name

class DFT9LayerTest : public testing::WithParamInterface<DFT9Params>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DFT9Params>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
