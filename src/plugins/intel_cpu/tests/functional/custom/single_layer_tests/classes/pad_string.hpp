// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

namespace ov {
namespace test {
namespace PadString {

// padsBegin, padsEnd, pad value string, input shape
using PadStringSpecificParams = std::tuple<
    std::vector<int64_t>,  // padsBegin
    std::vector<int64_t>,  // padsEnd
    std::string,           // pad value (constant mode)
    InputShape             // input shape
>;

using PadStringLayerTestParams = std::tuple<
    PadStringSpecificParams,
    ov::test::TargetDevice
>;

using PadStringLayerCPUTestParamsSet = std::tuple<
    PadStringLayerTestParams,
    CPUTestUtils::CPUSpecificParams
>;

class PadStringLayerCPUTest
    : public testing::WithParamInterface<PadStringLayerCPUTestParamsSet>,
      public SubgraphBaseTest,
      public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<PadStringLayerCPUTestParamsSet>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

extern const std::vector<PadStringSpecificParams> PadStringParamsVector;

}  // namespace PadString
}  // namespace test
}  // namespace ov
