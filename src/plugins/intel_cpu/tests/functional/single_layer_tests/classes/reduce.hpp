// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,               // Axis to reduce order
        ov::test::utils::OpType,        // Scalar or vector type axis
        bool,                           // Keep dims
        ngraph::helpers::ReductionType, // Reduce operation type
        ElementType,                    // Net precision
        ElementType,                    // Input precision
        ElementType,                    // Output precision
        std::vector<InputShape>         // Input shapes
> basicReduceParams;

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
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

private:
    ngraph::helpers::ReductionType reductionType;
    ElementType netPrecision;
};

namespace Reduce {

const std::vector<bool>& keepDims();
const std::vector<std::vector<int>>& axes();
const std::vector<std::vector<int>>& axesND();
const std::vector<ov::test::utils::OpType>& opTypes();
const std::vector<ngraph::helpers::ReductionType>& reductionTypes();
const std::vector<ElementType>& inpOutPrc();
const std::vector<std::map<std::string, ov::element::Type>> additionalConfig();
const std::vector<std::map<std::string, ov::element::Type>> additionalConfigFP32();
const std::vector<ngraph::helpers::ReductionType>& reductionTypesInt32();

} // namespace Reduce
} // namespace CPULayerTestsDefinitions
