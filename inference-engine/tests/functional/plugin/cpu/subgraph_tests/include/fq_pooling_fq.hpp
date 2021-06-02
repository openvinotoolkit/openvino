// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        std::vector<size_t>,             // Kernel size
        std::vector<size_t>,             // Strides
        std::vector<size_t>,             // Pad Begin
        std::vector<size_t>              // Pad end
> PoolingParams;

typedef std::tuple<
        InferenceEngine::SizeVector,     // Input shape
        PoolingParams,                   // Pooling params
        CPUSpecificParams
> FQPoolingFQCpuTestParamsSet;

class FQPoolingFQSubgraphTest : public testing::WithParamInterface<FQPoolingFQCpuTestParamsSet>, public CPUTestsBase,
                                virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FQPoolingFQCpuTestParamsSet> obj);

protected:
    void SetUp() override;
    void CheckFQCount(const size_t expectedFQCount);

private:
    InferenceEngine::SizeVector inputShape;
};

} // namespace SubgraphTestsDefinitions
