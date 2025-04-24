// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using ExtremumLayerCPUTestParamSet =
    std::tuple<std::vector<InputShape>,           // Input shapes
               utils::MinMaxOpType,               // Extremum type
               ov::element::Type,                 // Net precision
               ov::element::Type,                 // Input precision
               ov::element::Type,                 // Output precision
               CPUTestUtils::CPUSpecificParams,
               bool>;

class ExtremumLayerCPUTest : public testing::WithParamInterface<ExtremumLayerCPUTestParamSet>,
                             virtual public ov::test::SubgraphBaseTest,
                             public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExtremumLayerCPUTestParamSet> &obj);

protected:
    void SetUp() override;

private:
    std::string getPrimitiveType();
};

namespace extremum {

const std::vector<std::vector<ov::Shape>>& inputShape();

const std::vector<utils::MinMaxOpType>& extremumTypes();

const std::vector<ov::element::Type>& netPrecisions();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams4D();

}  // namespace extremum
}  // namespace test
}  // namespace ov
