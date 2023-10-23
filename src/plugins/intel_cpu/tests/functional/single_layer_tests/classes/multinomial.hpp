// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<InputShape,          // probs_shape
                InputShape,             // num_samples_shape
                ov::test::ElementType,  // convert_type
                bool,                   // with_replacement
                bool,                   // log_probs
                uint64_t,               // global_seed
                uint64_t,               // op_seed
> MultinomialCPUTestParams;

class MultinomialLayerCPUTest : public testing::WithParamInterface<MultinomialCPUTestParams>,,
                                public CPUTestsBase {
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformLayerTestCPUParamSet>& obj);

protected:
    void SetUp() override;
}
} // namespace CPULayerTestsDefinitions