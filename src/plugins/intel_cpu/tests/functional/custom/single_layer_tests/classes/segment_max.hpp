// Copyright (C) 2018-2025 Intel Corporation
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
namespace SegmentMax {
using SegmentMaxSpecificParams =  std::tuple<
        InputShape,
        std::vector<int64_t>,
        int64_t,
        ov::op::FillMode
>;

using SegmentMaxLayerTestParams = std::tuple<
        SegmentMaxSpecificParams,
        ElementType,
        ElementType,
        bool,
        ov::test::TargetDevice
>;

using SegmentMaxLayerCPUTestParamsSet = std::tuple<
        SegmentMaxLayerTestParams,
        CPUTestUtils::CPUSpecificParams>;

class SegmentMaxLayerCPUTest : public testing::WithParamInterface<SegmentMaxLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
   static std::string getTestCaseName(testing::TestParamInfo<SegmentMaxLayerCPUTestParamsSet> obj);
protected:
   void SetUp() override;
   void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

const std::vector<ElementType> indexPrecisions = {
        ElementType::i32,
        ElementType::i64
};

extern const std::vector<SegmentMaxSpecificParams> SegmentMaxParamsVector;
}  // namespace SegmentMax
}  // namespace test
}  // namespace ov
