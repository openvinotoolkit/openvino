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

using concatParamsTuple = typename std::tuple<
        //TODO: according to specification axis have to be int, negative values are allowed
        size_t,                            // Concat axis
        std::vector<std::vector<size_t>>,  // Input shapes
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class ConcatLayerTest : public testing::WithParamInterface<concatParamsTuple>,
                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions