// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/pooling.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using poolLayerCpuTestParamsSet = std::tuple<poolSpecificParams,
                                             InputShape,
                                             ElementType, //inPrc
                                             bool, // isInt8
                                             CPUSpecificParams,
                                             fusingSpecificParams>;

using maxPoolLayerCpuTestParamsSet = std::tuple<maxPoolSpecificParams,
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

class MaxPoolingV8LayerCPUTest : public testing::WithParamInterface<maxPoolLayerCpuTestParamsSet>,
                                 virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<maxPoolLayerCpuTestParamsSet>& obj);
    std::string getPrimitiveTypeUtil() const {
        return CPUTestsBase::getPrimitiveType();
    }
    void init_input_shapes_util(const std::vector<ov::test::InputShape> &shapes) {
        SubgraphBaseTest::init_input_shapes(shapes);
    }
protected:
    void SetUp() override;
};

class MaxPoolingV14LayerCPUTest : public testing::WithParamInterface<maxPoolLayerCpuTestParamsSet>,
                                  virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<maxPoolLayerCpuTestParamsSet>& obj);
    std::string getPrimitiveTypeUtil() const {
        return CPUTestsBase::getPrimitiveType();
    }
    void init_input_shapes_util(const std::vector<ov::test::InputShape> &shapes) {
        SubgraphBaseTest::init_input_shapes(shapes);
    }
protected:
    void SetUp() override;
};

namespace Pooling {
const std::vector<ElementType>& inpOutPrecision();
const ov::op::RoundingType expectedAvgRoundingType();

const std::vector<poolSpecificParams>& paramsMax3D();
const std::vector<poolSpecificParams>& paramsAvg3D();
const std::vector<poolSpecificParams>& paramsMax4D();

const std::vector<maxPoolSpecificParams>& paramsMaxV83D();
const std::vector<maxPoolSpecificParams>& paramsMaxV84D();
const std::vector<maxPoolSpecificParams>& paramsMaxV85D();

const std::vector<InputShape>& inputShapes3D();
const std::vector<InputShape>& inputShapes4D();
const std::vector<InputShape>& inputShapes4D_Large();
const std::vector<InputShape>& inputShapes5D();

const std::vector<poolSpecificParams>& paramsAvg4D();
const std::vector<poolSpecificParams>& paramsAvg4D_Large();
const std::vector<poolSpecificParams>& paramsAvg5D();
const std::vector<poolSpecificParams>& paramsMax5D();
}  // namespace Pooling
}  // namespace test
}  // namespace ov
