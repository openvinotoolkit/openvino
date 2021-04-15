// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using expParamsTuple = std::tuple<
    InferenceEngine::SizeVector,    // Input shape
    InferenceEngine::Precision,     // Net precision
    std::string>;                   // Device name

class ExpLayerTest : public testing::WithParamInterface<expParamsTuple>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<expParamsTuple> obj);
protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions