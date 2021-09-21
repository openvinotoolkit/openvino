// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::size_t,                        // Input Size
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string> //Configuration
> MatMulMultipleOutputsParams;

class MatMulMultipleOutputsTest:
        public testing::WithParamInterface<MatMulMultipleOutputsParams>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulMultipleOutputsParams> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions