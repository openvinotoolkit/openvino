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

class GatherLayerTestBase : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp(const gatherParamsTuple& params);
};

class GatherLayerTest : public testing::WithParamInterface<gatherParamsTuple>, public GatherLayerTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gatherParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
