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

using MaxMinParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, // Input shapes
        ngraph::helpers::MinMaxOpType,    // OperationType
        InferenceEngine::Precision,       // Network precision
        ngraph::helpers::InputLayerType,  // Secondary input type
        std::string>;                     // Device name

class MaxMinLayerTest:
        public testing::WithParamInterface<MaxMinParamsTuple>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxMinParamsTuple>& obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
