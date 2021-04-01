// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
    InferenceEngine::Precision,        //Network precision
    std::string,                       //Device name
    size_t,                            //Input size
    size_t,                            //Hidden size
    std::map<std::string, std::string> //Configuration
> ConcatQuantDuringMemoryRequantTuple;

class ConcatQuantDuringMemoryRequantTest
    : public testing::WithParamInterface<ConcatQuantDuringMemoryRequantTuple>,
    public LayerTestsUtils::LayerTestsCommon {
private:
    void switchToNgraphFriendlyModel();
    std::vector<float> memory_1_init;
    std::vector<float> memory_2_init;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatQuantDuringMemoryRequantTuple>& obj);
protected:
    void SetUp() override;
    void Run() override;
};
} // namespace SubgraphTestsDefinitions
