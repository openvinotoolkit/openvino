// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using InterpolateSpecificParams =
    std::tuple<ov::op::v11::Interpolate::InterpolateMode,          // InterpolateMode
               ov::op::v11::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
               ov::op::v11::Interpolate::NearestMode,              // NearestMode
               bool,                                               // AntiAlias
               std::vector<size_t>,                                // PadBegin
               std::vector<size_t>,                                // PadEnd
               double>;                                            // Cube coef

using ShapeParams = std::tuple<ov::op::v11::Interpolate::ShapeCalcMode,  // ShapeCalculationMode
                               InputShape,                               // Input shapes
                               // params describing input, choice of which depends on ShapeCalcMode
                               ov::test::utils::InputLayerType,  // input type
                               std::vector<std::vector<float>>,  // scales or sizes values
                               std::vector<int64_t>>;            // axes

using InterpolateLayerCPUTestParamsSet = std::tuple<InterpolateSpecificParams,
                                                    ShapeParams,
                                                    ElementType,
                                                    CPUSpecificParams,
                                                    fusingSpecificParams,
                                                    ov::AnyMap>;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void configure_model() override;

protected:
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> sizes;
    ov::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    size_t inferRequestNum = 0;

    void SetUp() override;
};

namespace Interpolate {
   const std::vector<ov::op::v11::Interpolate::NearestMode> defNearestModes();
   const std::vector<bool> antialias();
   const std::vector<double> cubeCoefs();
}  // namespace Interpolate
}  // namespace test
}  // namespace ov
