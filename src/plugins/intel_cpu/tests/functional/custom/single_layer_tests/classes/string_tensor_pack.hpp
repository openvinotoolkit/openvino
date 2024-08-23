// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

namespace ov {
namespace test {
namespace StringTensorPack {
using StringTensorPackSpecificParams =  std::tuple<
        InputShape,                     // begins/ends shape
        InputShape                      // utf-8 encoded symbols shape
>;

using StringTensorPackLayerTestParams = std::tuple<
        StringTensorPackSpecificParams,
        ElementType,
        ov::test::TargetDevice
>;

using StringTensorPackLayerCPUTestParamsSet = std::tuple<
        StringTensorPackLayerTestParams,
        CPUTestUtils::CPUSpecificParams>;

class StringTensorPackLayerCPUTest : public testing::WithParamInterface<StringTensorPackLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
   static std::string getTestCaseName(testing::TestParamInfo<StringTensorPackLayerCPUTestParamsSet> obj);
protected:
   void SetUp() override;
   void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

extern const std::vector<StringTensorPackSpecificParams> StringTensorPackParamsVector;
}  // namespace StringTensorPack
}  // namespace test
}  // namespace ov
