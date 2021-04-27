// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using constantParamsTuple = typename std::tuple<
    std::vector<size_t>,         // Constant data shape
    InferenceEngine::Precision,  // Constant data precision
    std::vector<std::string>,    // Constant elements
    std::string>;                // Device name

class ConstantLayerTest : public testing::WithParamInterface<constantParamsTuple>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<constantParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
