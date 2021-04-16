// Copyright (C) 2018-2021 Intel Corporation
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
#include <ngraph/opsets/opset7.hpp>

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,                  // Indices
        std::vector<size_t>,               // Indices shape
        int,                               // Gather axis
        std::vector<size_t>,               // Input shapes
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string                        // Device name
> gatherParamsTuple;

typedef std::tuple<
        gatherParamsTuple,
        int                                // Gather batch_dim
> gather7ParamsTuple;

class Gather7LayerTestBase : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp(const gather7ParamsTuple& params);
};

class Gather7LayerTest : public testing::WithParamInterface<gather7ParamsTuple>, public Gather7LayerTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions