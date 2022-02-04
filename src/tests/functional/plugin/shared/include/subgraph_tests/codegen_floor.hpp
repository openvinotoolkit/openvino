// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input 0 Shape
        std::string                  // Target Device
> inputParams;

class CodegenFloor : public testing::WithParamInterface<LayerTestsDefinitions::inputParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj);

protected:
    void GenerateInputs() override;
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
