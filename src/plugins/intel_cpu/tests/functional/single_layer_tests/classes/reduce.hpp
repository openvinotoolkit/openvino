// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,                   // Axis to reduce order
        CommonTestUtils::OpType,            // Scalar or vector type axis
        bool,                               // Keep dims
        ngraph::helpers::ReductionType,     // Reduce operation type
        ov::test::ElementType,              // Net precision
        ov::test::ElementType,              // Input precision
        ov::test::ElementType,              // Output precision
        std::vector<ov::test::InputShape>,  // Input shapes
        ov::AnyMap                          // Additional network configuration
> basicReduceParams;

typedef std::tuple<
        basicReduceParams,
        CPUTestUtils::CPUSpecificParams,
        CPUTestUtils::fusingSpecificParams> ReduceLayerCPUTestParamSet;

class ReduceCPULayerTest : public testing::WithParamInterface<ReduceLayerCPUTestParamSet>,
                           virtual public ov::test::SubgraphBaseTest, public CPUTestUtils::CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj);
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

private:
    ngraph::helpers::ReductionType reductionType;
    ov::test::ElementType netPrecision;
};

namespace Reduce {

const std::vector<bool>& keepDims();
const std::vector<std::vector<int>>& axes();
const std::vector<std::vector<int>>& axesND();
const std::vector<CommonTestUtils::OpType>& opTypes();
const std::vector<ngraph::helpers::ReductionType>& reductionTypes();
const std::vector<ov::test::ElementType>& inpOutPrc();
const std::vector<ngraph::helpers::ReductionType>& reductionTypesInt32();

} // namespace Reduce
} // namespace CPULayerTestsDefinitions
