// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace ov {
namespace test {

typedef std::tuple<std::vector<int>,         // Axis to reduce order
                   ov::test::utils::OpType,  // Scalar or vector type axis
                   bool,                     // Keep dims
                   utils::ReductionType,     // Reduce operation type
                   ElementType,              // Net precision
                   ElementType,              // Input precision
                   ElementType,              // Output precision
                   std::vector<InputShape>   // Input shapes
                   >
    basicReduceParams;

typedef std::tuple<
        basicReduceParams,
        CPUSpecificParams,
        fusingSpecificParams,
        std::map<std::string, ov::element::Type>> ReduceLayerCPUTestParamSet;

class ReduceCPULayerTest : public testing::WithParamInterface<ReduceLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj);
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

private:
    utils::ReductionType reductionType;
    ElementType netPrecision;
};

namespace Reduce {

const std::vector<bool>& keepDims();
const std::vector<std::vector<int>>& axes();
const std::vector<std::vector<int>>& axesND();
const std::vector<ov::test::utils::OpType>& opTypes();
const std::vector<utils::ReductionType>& reductionTypes();
const std::vector<ElementType>& inpOutPrc();
const std::vector<std::map<std::string, ov::element::Type>> additionalConfig();
const std::vector<std::map<std::string, ov::element::Type>> additionalConfigFP32();
const std::vector<utils::ReductionType>& reductionTypesInt32();
const std::vector<utils::ReductionType>& reductionTypesNativeInt32();

}  // namespace Reduce
}  // namespace test
}  // namespace ov
