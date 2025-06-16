// Copyright (C) 2018-2025 Intel Corporation
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
namespace Col2Im {
using Col2ImSpecificParams =  std::tuple<
        InputShape,                                         // data shape
        std::vector<int64_t>,                               // output size values
        std::vector<int64_t>,                               // kernel size values
        ov::Strides,                                        // strides
        ov::Strides,                                        // dilations
        ov::Shape,                                          // pads_begin
        ov::Shape                                           // pads_end
>;

using Col2ImLayerTestParams = std::tuple<
        Col2ImSpecificParams,
        ElementType,                                        // data precision
        ElementType,                                        // index precision
        ov::test::TargetDevice                              // device name
>;

using Col2ImLayerCPUTestParamsSet = std::tuple<
        Col2ImLayerTestParams,
        CPUSpecificParams>;

class Col2ImLayerCPUTest : public testing::WithParamInterface<Col2ImLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestsBase {
public:
   static std::string getTestCaseName(testing::TestParamInfo<Col2ImLayerCPUTestParamsSet> obj);
protected:
   void SetUp() override;
   void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

const std::vector<ElementType> indexPrecisions = {
        ElementType::i32,
        ElementType::i64
};

extern const std::vector<Col2ImSpecificParams> col2ImParamsVector;
}  // namespace Col2Im
}  // namespace test
}  // namespace ov
