// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        size_t,
        size_t,
        InferenceEngine::Precision,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        std::string> splitParams;

class SplitLayerTest
        : public LayerTestsUtils::LayerTestsCommonClass<splitParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions