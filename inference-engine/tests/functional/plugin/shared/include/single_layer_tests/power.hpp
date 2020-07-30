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
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

    using PowerParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, //input shapes
        InferenceEngine::Precision,       //Network precision
        std::string,                      //Device name
        std::vector<float>>;               //power

class PowerLayerTest:
        public testing::WithParamInterface<PowerParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PowerParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
