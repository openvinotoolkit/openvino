// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
namespace SubtractTestDefinitions {
    enum class SecondaryInputType {
        CONSTANT,
        PARAMETER
    };
    enum class SubtractionType {
        SCALAR,
        VECTOR
    };
    const char* SecondaryInputType_to_string(SecondaryInputType input_type);
    const char* SubtractionType_to_string(SubtractionType subtraction_type);
} // namespace SubtractTestDefinitions

using SubtractParamsTuple = typename std::tuple<
    std::vector<std::vector<size_t>>,             // input shapes
    SubtractTestDefinitions::SecondaryInputType,  // type of secondary input node
    SubtractTestDefinitions::SubtractionType,     // type of multiplication (vector, scalar)
    InferenceEngine::Precision,                   // Network precision
    std::string,                                  // Device name
    std::map<std::string, std::string>>;          // Additional network configuration

class SubtractLayerTest : public testing::WithParamInterface<SubtractParamsTuple>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SubtractParamsTuple> obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
