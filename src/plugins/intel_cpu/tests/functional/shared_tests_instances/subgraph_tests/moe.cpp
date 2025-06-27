// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest-param-test.h>
#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/subgraph/moe_pattern.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov::test {
using namespace CPUTestUtils;

TEST_P(MOETest, Inference) {
    // only implement on x86
#if !defined(OPENVINO_ARCH_X86) && !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif

    targetDevice = ov::test::utils::DEVICE_CPU;
    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "MOE", 1);
    CheckNumberOfNodesWithType(compiledModel, "OneHot", 0);
    auto expectedOutputs = run_test(functionRefs);

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_MOE_basic,
                         MOETest,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::bf16, ov::element::f32),
                            ::testing::Values(ov::element::f16, ov::element::u8, ov::element::u4)),
                         MOETest::getTestCaseName);

} // namespace ov::test
