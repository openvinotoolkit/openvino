// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "functional_test_utils/layer_test_utils.hpp"
#include "../../../../../ngraph_functions/include/ngraph_functions/builders.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

using ScaleShiftParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, //input shapes
        InferenceEngine::Precision,       //Network precision
        std::string,                      //Device name
        std::vector<float>,               //scale
        std::vector<float>>;              //shift

class ScaleShiftLayerTest:
        public testing::WithParamInterface<ScaleShiftParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScaleShiftParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
