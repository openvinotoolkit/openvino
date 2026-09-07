// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "cache/multi_cache.h"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::test {

class TestKernelConfig : public ov::snippets::KernelExecutorBase::GenericConfig {
public:
    bool is_completed() const override {
        return true;
    }

    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::make_unique<TestKernelConfig>();
    }

    size_t hash() const override {
        return 0;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override {
        return "TestKernelConfig";
    }
#endif
};

class TestKernelExecutor : public ov::snippets::KernelExecutor<TestKernelConfig, int> {
public:
    TestKernelExecutor() : ov::snippets::KernelExecutor<TestKernelConfig, int>(TestKernelConfig{}) {}

protected:
    void update_config(const ov::snippets::lowered::ExpressionPtr&,
                       const ov::snippets::lowered::LinearIRCPtr&,
                       TestKernelConfig&) const override {}

    void update_kernel(const TestKernelConfig&, std::shared_ptr<int>&) const override {
    }
};

TEST(CPURuntimeConfig, CopyCreatesIndependentKernelExecutorTable) {
    CPURuntimeConfig original;
    original.kernel_executor_table = std::make_shared<ov::snippets::KernelExecutorTable>();
    const auto expression = std::make_shared<ov::snippets::lowered::Expression>();
    const auto executor = original.kernel_executor_table->register_kernel<TestKernelExecutor>(expression);

    const CPURuntimeConfig cloned(original);

    EXPECT_NE(original.kernel_executor_table, cloned.kernel_executor_table);
    EXPECT_NE(nullptr, cloned.kernel_executor_table);
    EXPECT_EQ(executor, original.kernel_executor_table->get_kernel_executor(expression));
}

TEST(CPURuntimeConfigurator, CopyCreatesIndependentKernelExecutorTable) {
    const auto cache = std::make_shared<MultiCache>(1);
    CPURuntimeConfigurator original(cache);
    const auto original_table = std::make_shared<ov::snippets::KernelExecutorTable>();
    original.set_kernel_executor_table(original_table);
    const auto expression = std::make_shared<ov::snippets::lowered::Expression>();
    const auto executor = original_table->register_kernel<TestKernelExecutor>(expression);

    CPURuntimeConfigurator cloned(original);
    const auto cloned_table = cloned.get_kernel_executor_table();

    EXPECT_NE(original.get_kernel_executor_table(), cloned_table);
    EXPECT_NE(nullptr, cloned_table);
    EXPECT_EQ(executor, original.get_kernel_executor_table()->get_kernel_executor(expression));

    original.reset_kernel_executor_table();
    EXPECT_EQ(cloned_table, cloned.get_kernel_executor_table());
    EXPECT_NE(original.get_kernel_executor_table(), cloned.get_kernel_executor_table());

    const auto replacement_cloned_table = std::make_shared<ov::snippets::KernelExecutorTable>();
    cloned.set_kernel_executor_table(replacement_cloned_table);

    EXPECT_EQ(replacement_cloned_table, cloned.get_kernel_executor_table());
    EXPECT_NE(original.get_kernel_executor_table(), cloned.get_kernel_executor_table());
}

}  // namespace ov::intel_cpu::test
