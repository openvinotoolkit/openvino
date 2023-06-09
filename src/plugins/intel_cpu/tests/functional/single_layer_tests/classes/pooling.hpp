// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/single_layer/pooling.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using poolLayerCpuTestParamsSet = std::tuple<LayerTestsDefinitions::poolSpecificParams,
                                             InputShape,
                                             ElementType,
                                             bool,
                                             CPUSpecificParams,
                                             fusingSpecificParams>;

using maxPoolV8LayerCpuTestParamsSet = std::tuple<LayerTestsDefinitions::maxPoolV8SpecificParams,
        InputShape,
        ElementType,
        CPUSpecificParams>;

class PoolingLayerCPUTest : public testing::WithParamInterface<poolLayerCpuTestParamsSet>,
                            virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj);

protected:
    void SetUp() override;
};

class MaxPoolingV8LayerCPUTest : public testing::WithParamInterface<maxPoolV8LayerCpuTestParamsSet>,
                                 virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<maxPoolV8LayerCpuTestParamsSet>& obj);

protected:
    void SetUp() override;
};

namespace Pooling {

} // namespace Pooling
} // namespace CPULayerTestsDefinitions