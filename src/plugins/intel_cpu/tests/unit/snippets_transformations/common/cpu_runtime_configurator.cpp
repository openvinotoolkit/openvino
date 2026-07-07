// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "cache/multi_cache.h"
#include "snippets/kernel_executor_table.hpp"

namespace ov::intel_cpu::test {

TEST(CPURuntimeConfig, CopyCreatesIndependentKernelExecutorTable) {
    CPURuntimeConfig original;
    original.kernel_executor_table = std::make_shared<ov::snippets::KernelExecutorTable>();

    const CPURuntimeConfig cloned(original);

    EXPECT_NE(original.kernel_executor_table, cloned.kernel_executor_table);
    EXPECT_NE(nullptr, cloned.kernel_executor_table);
}

TEST(CPURuntimeConfigurator, CopyCreatesIndependentKernelExecutorTable) {
    const auto cache = std::make_shared<MultiCache>(1);
    CPURuntimeConfigurator original(cache);
    const auto original_table = std::make_shared<ov::snippets::KernelExecutorTable>();
    original.set_kernel_executor_table(original_table);

    CPURuntimeConfigurator cloned(original);
    const auto cloned_table = cloned.get_kernel_executor_table();

    EXPECT_NE(original.get_kernel_executor_table(), cloned_table);
    EXPECT_NE(nullptr, cloned_table);

    original.reset_kernel_executor_table();
    EXPECT_EQ(cloned_table, cloned.get_kernel_executor_table());
    EXPECT_NE(original.get_kernel_executor_table(), cloned.get_kernel_executor_table());

    const auto replacement_cloned_table = std::make_shared<ov::snippets::KernelExecutorTable>();
    cloned.set_kernel_executor_table(replacement_cloned_table);

    EXPECT_EQ(replacement_cloned_table, cloned.get_kernel_executor_table());
    EXPECT_NE(original.get_kernel_executor_table(), cloned.get_kernel_executor_table());
}

}  // namespace ov::intel_cpu::test
