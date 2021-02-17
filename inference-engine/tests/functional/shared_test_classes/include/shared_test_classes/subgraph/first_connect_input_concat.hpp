// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include <ngraph_functions/builders.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>


namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,   // Input shapes
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  // Config
> concatFirstInputParams;

class ConcatFirstInputTest : public testing::WithParamInterface<concatFirstInputParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatFirstInputParams> obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
