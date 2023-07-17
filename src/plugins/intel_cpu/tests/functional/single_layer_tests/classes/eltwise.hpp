// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
        ov::test::subgraph::EltwiseTestParams,
        CPUTestUtils::CPUSpecificParams,
        CPUTestUtils::fusingSpecificParams> EltwiseLayerCPUTestParamsSet;

class EltwiseLayerCPUTest : public testing::WithParamInterface<EltwiseLayerCPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest,
                            public CPUTestUtils::CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj);

protected:
    ov::Tensor generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape, size_t in_idx = 0llu);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

private:
    ngraph::helpers::EltwiseTypes eltwiseType;
};

namespace Eltwise {

const ov::AnyMap& additional_config();

const std::vector<ov::test::ElementType>& netType();
const std::vector<ov::test::utils::OpType>& opTypes();
const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinInp();
const std::vector<ngraph::helpers::InputLayerType>& secondaryInputTypes();

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinInp();
const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesDiffInp();
const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinDyn();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams_4D();
const std::vector<std::vector<ov::Shape>>& inShapes_4D();
const std::vector<std::vector<ov::test::InputShape>>& inShapes_4D_dyn_const();
const std::vector<ov::test::InputShape>& inShapes_4D_dyn_param();
const std::vector<std::vector<ov::Shape>>& inShapes_4D_1D();
const std::vector<CPUTestUtils::CPUSpecificParams> & cpuParams_4D_1D_Constant_mode();
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams_4D_1D_Parameter_mode();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams_5D();
const std::vector<std::vector<ov::Shape>>& inShapes_5D();
const std::vector<std::vector<ov::Shape>>& inShapes_5D_1D();
const std::vector<ov::test::InputShape>& inShapes_5D_dyn_const();
const std::vector<ov::test::InputShape>& inShapes_5D_dyn_param();
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams_5D_1D_constant();
const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams_5D_1D_parameter();

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesI32();

} // namespace Eltwise
} // namespace CPULayerTestsDefinitions
