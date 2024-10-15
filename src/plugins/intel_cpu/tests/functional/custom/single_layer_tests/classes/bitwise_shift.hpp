// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<EltwiseTestParams, CPUSpecificParams, fusingSpecificParams, bool, ov::AnyMap>
    BitshiftLayerCPUTestParamsSet;

class BitwiseShiftLayerCPUTest : public testing::WithParamInterface<BitshiftLayerCPUTestParamsSet>,
                                 virtual public SubgraphBaseTest,
                                 public CPUTestUtils::CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BitshiftLayerCPUTestParamsSet> obj);

    ov::Tensor generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape, size_t in_idx = 0);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

private:
    utils::EltwiseTypes eltwiseType;
    std::shared_ptr<ov::op::v0::Constant> shift_const;
};
}  // namespace test
}  // namespace ov
