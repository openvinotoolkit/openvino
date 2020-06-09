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

using MultiplyAddParamsTuple = typename std::tuple<
        std::vector<size_t>,              //input shapes
        InferenceEngine::Precision,       //Network precision
        std::string>;                     //Device name

class MultiplyAddLayerTest:
        public testing::WithParamInterface<MultiplyAddParamsTuple>,
        public LayerTestsUtils::LayerTestsCommon{
public:
    std::shared_ptr<ngraph::Function> fn;
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyAddParamsTuple> &obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
