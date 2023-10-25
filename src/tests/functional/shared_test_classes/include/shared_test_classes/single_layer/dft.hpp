// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::SizeVector, // Input shapes
        InferenceEngine::Precision,  // Input precision
        std::vector<int64_t>,  // Axes
        std::vector<int64_t>,  // Signal size
        ngraph::helpers::DFTOpType,
        std::string> DFTParams;   // Device name

class DFTLayerTest : public testing::WithParamInterface<DFTParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DFTParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
