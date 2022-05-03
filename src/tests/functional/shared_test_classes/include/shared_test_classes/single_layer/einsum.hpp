// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::string,                       // Equation
        std::vector<std::vector<size_t>>   // Input shapes
> EinsumEquationWithInput;

typedef std::tuple<
        InferenceEngine::Precision,         // Input precision
        EinsumEquationWithInput,            // Equation with corresponding input shapes
        std::string                         // Device name
> EinsumLayerTestParamsSet;

class EinsumLayerTest : public testing::WithParamInterface<EinsumLayerTestParamsSet>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EinsumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
