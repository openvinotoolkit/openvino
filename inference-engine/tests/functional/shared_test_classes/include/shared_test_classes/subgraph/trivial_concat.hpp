// Copyright (C) 2019 Intel Corporation
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

namespace SubgraphTestsDefinitions {
using trivialConcatParamsTuple = typename std::tuple<
    std::vector<size_t>,               // Inputs shape
    InferenceEngine::Precision,        // Network precision
    std::string,                       // Device name
    std::map<std::string, std::string> // Configuration
>;

class TrivialConcatLayerTest : public testing::WithParamInterface<trivialConcatParamsTuple>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<trivialConcatParamsTuple> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
