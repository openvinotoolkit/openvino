// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov::intel_cpu::aarch64 {

class ParallelLoopConfig : public ov::snippets::KernelExecutorBase::GenericConfig {
public:
    ParallelLoopConfig() = default;
    explicit ParallelLoopConfig(size_t increment) : m_increment(increment) {}

    [[nodiscard]] bool is_completed() const override {
        return true;
    }

    [[nodiscard]] std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::make_unique<ParallelLoopConfig>(*this);
    }

    [[nodiscard]] size_t hash() const override;

    [[nodiscard]] size_t get_increment() const {
        return m_increment;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    [[nodiscard]] std::string to_string() const override {
        return "increment = " + std::to_string(m_increment);
    }
#endif

protected:
    size_t m_increment = 0;
};

class ParallelLoopKernel {};

class ParallelLoopExecutor : public snippets::KernelExecutor<ParallelLoopConfig, ParallelLoopKernel> {
public:
    using loop_preamble_t = void (*)(int64_t, void*);
    struct call_args {
        jit_snippets_call_args::loop_args_t* loop_args = nullptr;
        loop_preamble_t preamble_ptr = nullptr;
        uintptr_t** mem_ptrs = nullptr;
    };

    explicit ParallelLoopExecutor(ParallelLoopConfig config) : KernelExecutor(std::move(config)) {}

    static void execute(const ParallelLoopExecutor* executor, call_args* call_args);

protected:
    void update_config(const snippets::lowered::ExpressionPtr& expr,
                       const snippets::lowered::LinearIRCPtr& linear_ir,
                       ParallelLoopConfig& config) const override {}
    void update_kernel(const ParallelLoopConfig& config, std::shared_ptr<ParallelLoopKernel>& kernel) const override;
};

#define GET_OFF_PARALLEL_LOOP_ARGS(field) offsetof(ParallelLoopExecutor::call_args, field)

}  // namespace ov::intel_cpu::aarch64
