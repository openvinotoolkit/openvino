// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/mvn.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"


using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using basicCpuMvnParams = std::tuple<
       InputShape, // Input shapes
       ElementType, // Input precision
       ngraph::AxisSet, // Reduction axes
       bool, // Across channels
       bool, // Normalize variance
       double>; // Epsilon

using MvnLayerCPUTestParamSet = std::tuple<
       basicCpuMvnParams,
       CPUSpecificParams,
       fusingSpecificParams,
       ElementType, // CNNNetwork input precision
       ElementType, // CNNNetwork output precision
       std::map<std::string, ov::element::Type>>;

class MvnLayerCPUTest : public testing::WithParamInterface<MvnLayerCPUTestParamSet>,
                       virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
   static std::string getTestCaseName(testing::TestParamInfo<MvnLayerCPUTestParamSet> obj);
   bool isSupportedTestCase();
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

   const std::vector<ngraph::AxisSet>& emptyReductionAxes();
   const std::vector<bool>& acrossChannels();
   const std::vector<double>& epsilon();

   const std::vector<std::map<std::string, ov::element::Type>>& additionalConfig();
} // namespace MVN
} // namespace CPULayerTestsDefinitions