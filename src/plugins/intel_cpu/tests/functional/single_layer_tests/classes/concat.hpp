// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/concat.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        size_t,                   // Concat axis
        std::vector<InputShape>,  // Input shapes
        ElementType,              // Network precision
        CPUSpecificParams
> concatCPUTestParams;

class ConcatLayerCPUTest : public testing::WithParamInterface<concatCPUTestParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatCPUTestParams> obj);
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;

protected:
    size_t inferNum = 0;
    void SetUp() override;
};

namespace Concat {

} // namespace Concat
} // namespace CPULayerTestsDefinitions