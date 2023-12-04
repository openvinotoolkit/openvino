// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

using MaxMinParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, // Input shapes
        ngraph::helpers::MinMaxOpType,    // OperationType
        InferenceEngine::Precision,       // Network precision
        InferenceEngine::Precision,       // Input precision
        InferenceEngine::Precision,       // Output precision
        InferenceEngine::Layout,          // Input layout
        InferenceEngine::Layout,          // Output layout
        ngraph::helpers::InputLayerType,  // Secondary input type
        std::string>;                     // Device name

class MaxMinLayerTest:
        public testing::WithParamInterface<MaxMinParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MaxMinParamsTuple>& obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
