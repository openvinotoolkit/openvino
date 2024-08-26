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
namespace StringTensorUnpack {
using StringTensorUnpackSpecificParams =  std::tuple<
        InputShape
>;

using StringTensorUnpackLayerTestParams = std::tuple<
        StringTensorUnpackSpecificParams,
        ov::test::TargetDevice
>;

using StringTensorUnpackLayerCPUTestParamsSet = std::tuple<
        StringTensorUnpackLayerTestParams,
        CPUTestUtils::CPUSpecificParams>;

class StringTensorUnpackLayerCPUTest : public testing::WithParamInterface<StringTensorUnpackLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
   static std::string getTestCaseName(testing::TestParamInfo<StringTensorUnpackLayerCPUTestParamsSet> obj);
protected:
   void SetUp() override;
   void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

extern const std::vector<StringTensorUnpackSpecificParams> StringTensorUnpackParamsVector;
}  // namespace StringTensorUnpack
}  // namespace test
}  // namespace ov
