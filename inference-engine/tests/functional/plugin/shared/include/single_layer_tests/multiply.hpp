// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {
namespace MultiplyTestDefinitions {
    enum class SecondaryInputType {
        CONSTANT,
        PARAMETER
    };
    enum class MultiplicationType {
        SCALAR,
        VECTOR
    };
    const char* SecondaryInputType_to_string(SecondaryInputType input_type);
    const char* MultiplicationType_to_string(MultiplicationType multiplication_type);
} // namespace MultiplyTestDefinitions

using MultiplyParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>,             // input shapes
        MultiplyTestDefinitions::SecondaryInputType,  // type of secondary input node
        MultiplyTestDefinitions::MultiplicationType,  // type of multiplication (vector, scalar)
        InferenceEngine::Precision,                   // Network precision
        std::string,                                  // Device name
        std::map<std::string, std::string>>;          // Additional network configuration

class MultiplyLayerTest:
        public testing::WithParamInterface<MultiplyParamsTuple>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    std::shared_ptr<ngraph::Function> fn;
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
