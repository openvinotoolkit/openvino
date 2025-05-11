// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using basicCpuMvnParams = std::tuple<
       InputShape, // Input shapes
       ElementType, // Input precision
       ov::AxisSet, // Reduction axes
       bool, // Across channels
       bool, // Normalize variance
       double>; // Epsilon

using MvnLayerCPUTestParamSet = std::tuple<
       basicCpuMvnParams,
       CPUSpecificParams,
       fusingSpecificParams,
       ElementType, // model input precision
       ElementType, // model output precision
       ov::AnyMap>;

class MvnLayerCPUTest : public testing::WithParamInterface<MvnLayerCPUTestParamSet>,
                       virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
   static std::string getTestCaseName(testing::TestParamInfo<MvnLayerCPUTestParamSet> obj);
protected:
   void SetUp() override;
private:
   bool acrossChanels;
};

namespace MVN {
   const std::vector<InputShape>& inputShapes_1D();
   const std::vector<InputShape>& inputShapes_2D();
   const std::vector<InputShape>& inputShapes_3D();
   const std::vector<InputShape>& inputShapes_4D();
   const std::vector<InputShape>& inputShapes_5D();

   const std::vector<ov::Shape>& inputShapesStatic_2D();
   const std::vector<ov::Shape>& inputShapesStatic_3D();
   const std::vector<ov::Shape>& inputShapesStatic_4D();
   const std::vector<ov::Shape>& inputShapesStatic_5D();

   const std::vector<ov::AxisSet>& emptyReductionAxes();
   const std::vector<bool>& acrossChannels();
   const std::vector<double>& epsilon();

   const std::vector<CPUSpecificParams>& cpuParams_4D();
   const std::vector<CPUSpecificParams>& cpuParams_5D();

   const std::vector<ElementType>& inpPrc();
   const std::vector<ElementType>& outPrc();

   const std::vector<fusingSpecificParams>& fusingParamsSet();
   const std::vector<ov::AnyMap>& additionalConfig();
}  // namespace MVN
}  // namespace test
}  // namespace ov