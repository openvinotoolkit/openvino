// Copyright (C) 2019 Intel Corporation
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

namespace LayerTestsDefinitions {

using ConvertLikeParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>,  // Input1 shapes
        InferenceEngine::Precision,        // Input1 precision
        std::vector<std::vector<size_t>>,  // Input2 shapes
        InferenceEngine::Precision,        // Input2 precision
        std::string>;                      // Device name

class ConvertLikeLayerTest : public testing::WithParamInterface<ConvertLikeParamsTuple>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertLikeParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions