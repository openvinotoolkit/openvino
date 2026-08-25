// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>

#include "nodes/executors/implementations.hpp"

namespace ov::intel_cpu::test {
namespace {

template <typename Attributes>
void expect_jit_before_common(const char* jit_name, const char* common_name, OperationType operation_type) {
    const auto& implementations = getImplementations<Attributes>();
    ASSERT_EQ(implementations.size(), 2U);

    EXPECT_STREQ(implementations[0].name(), jit_name);
    EXPECT_EQ(implementations[0].type(), ExecutorType::Jit);
    EXPECT_EQ(implementations[0].operationType(), operation_type);

    EXPECT_STREQ(implementations[1].name(), common_name);
    EXPECT_EQ(implementations[1].type(), ExecutorType::Common);
    EXPECT_EQ(implementations[1].operationType(), operation_type);
}

TEST(SelectiveSSMJitExecutor, IsRegisteredBeforePortableFallback) {
    expect_jit_before_common<SelectiveSSMAttrs>("selective_ssm_jit_executor",
                                                "selective_ssm_common_executor",
                                                OperationType::SelectiveSSM);
}

TEST(PagedSelectiveSSMJitExecutor, IsRegisteredBeforePortableFallback) {
    expect_jit_before_common<PagedSelectiveSSMAttrs>("paged_selective_ssm_jit_executor",
                                                     "paged_selective_ssm_common_executor",
                                                     OperationType::PagedSelectiveSSM);
}

}  // namespace
}  // namespace ov::intel_cpu::test
