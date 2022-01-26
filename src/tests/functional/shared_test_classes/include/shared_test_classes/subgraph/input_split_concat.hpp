// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        std::vector<size_t>                 // Input Shapes
> InputSplitConcatParams;

class InputSplitConcatTest : public testing::WithParamInterface<InputSplitConcatParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InputSplitConcatParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
