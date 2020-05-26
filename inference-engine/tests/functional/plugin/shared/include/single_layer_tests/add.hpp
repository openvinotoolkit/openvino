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
namespace AddTestDefinitions {
    enum class SecondaryInputType {
        CONSTANT,
        PARAMETER
    };
    enum class AdditionType {
        SCALAR,
        VECTOR
    };
    const char* SecondaryInputType_to_string(SecondaryInputType input_type);
    const char* AdditionType_to_string(AdditionType multiplication_type);
} // namespace AddTestDefinitions

using AddParamsTuple = typename std::tuple<
    std::vector<std::vector<size_t>>,        // input shapes
    AddTestDefinitions::SecondaryInputType,  // type of secondary input node
    AddTestDefinitions::AdditionType,        // type of multiplication (vector, scalar)
    InferenceEngine::Precision,              // Network precision
    std::string,                             // Device name
    std::map<std::string, std::string>>;     // Additional network configuration

class AddLayerTest : public testing::WithParamInterface<AddParamsTuple>,
                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AddParamsTuple> obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
