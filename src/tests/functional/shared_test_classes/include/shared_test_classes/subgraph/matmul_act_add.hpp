// Copyright (C) 2018-2022 Intel Corporation
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
> MatMulActAddParams;

class MatMulActAddTest:
        public testing::WithParamInterface<MatMulActAddParams>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulActAddParams> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions