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
        std::vector<size_t>                  // Input Shapes
> softsignParams;

class SoftsignTest : public testing::WithParamInterface<softsignParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softsignParams>& obj);

    void Run() override;

protected:
    void SetUp() override;

private:
    std::shared_ptr<ngraph::Function> GenerateNgraphFriendlySoftSign();
};

}  // namespace SubgraphTestsDefinitions
