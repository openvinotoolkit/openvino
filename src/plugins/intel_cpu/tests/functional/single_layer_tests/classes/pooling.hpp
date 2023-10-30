// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_models/builders.hpp"
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
                                             fusingSpecificParams,
                                             std::map<std::string, std::string>>;


using maxPoolV8LayerCpuTestParamsSet = std::tuple<LayerTestsDefinitions::maxPoolV8SpecificParams,
        InputShape,
        ElementType,
        CPUSpecificParams,
        std::map<std::string, std::string>>;

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
const std::vector<ElementType>& inpOutPrecision();
const ngraph::op::RoundingType expectedAvgRoundingType();

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsMax3D();
const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg3D();
const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsMax4D();

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams>& paramsMaxV84D();
const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams>& paramsMaxV85D();

const std::vector<InputShape>& inputShapes3D();
const std::vector<InputShape>& inputShapes4D();
const std::vector<InputShape>& inputShapes4D_Large();
const std::vector<InputShape>& inputShapes5D();

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg4D();
const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg4D_Large();
const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg5D();
const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsMax5D();
} // namespace Pooling
} // namespace CPULayerTestsDefinitions
