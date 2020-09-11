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

using ConvertParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>,  // Input shapes
        InferenceEngine::Precision,        // Source precision
        InferenceEngine::Precision,        // Target precision
        std::string>;                      // Device name

class ConvertLayerTest : public testing::WithParamInterface<ConvertParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions