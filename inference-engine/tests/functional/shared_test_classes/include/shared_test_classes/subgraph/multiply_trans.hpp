// Copyright (C) 2020 Intel Corporation
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

using MultiplyTransParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>,              //trans order
        InferenceEngine::Precision,       //Network precision
        std::string>;                     //Device name

class MultiplyTransLayerTest:
        public testing::WithParamInterface<MultiplyTransParamsTuple>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    std::shared_ptr<ngraph::Function> fn;
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyTransParamsTuple> &obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
