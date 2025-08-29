// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov::intel_cpu {

class ParallelLoopConfig : public ov::snippets::KernelExecutorBase::GenericConfig {
public:
    ParallelLoopConfig() = default;
    ParallelLoopConfig(size_t increment) : m_increment(increment) {}

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
    };
#endif

protected:
    size_t m_increment = 0;
};

// Note: the ParallelLoopKernel is empty because this executor doesn't need a kernel
class ParallelLoopKernel {};

class ParallelLoopExecutor : public snippets::KernelExecutor<ParallelLoopConfig, ParallelLoopKernel> {
public:
    // This function is used to initialize per-thread loop begin
    // Parameters:
    // - work_amount: work amount that should be handled in the current thread
    // - mem_ptrs: memory pointers with applied ptr_increments for the current thread
    using loop_preamble_t = void (*)(int64_t, void*);
    struct call_args {
        jit_snippets_call_args::loop_args_t* loop_args = nullptr;
        loop_preamble_t preamble_ptr = nullptr;
        uintptr_t** mem_ptrs = nullptr;
    };
    ParallelLoopExecutor(ParallelLoopConfig config) : KernelExecutor(std::move(config)) {}

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const ParallelLoopExecutor* executor, call_args* call_args);

protected:
    void update_config(const snippets::lowered::ExpressionPtr& expr,
                       const snippets::lowered::LinearIRCPtr& linear_ir,
                       ParallelLoopConfig& config) const override {}
    void update_kernel(const ParallelLoopConfig& c, std::shared_ptr<ParallelLoopKernel>& kernel) const override;
};
#define GET_OFF_PARALLEL_LOOP_ARGS(field) offsetof(ParallelLoopExecutor::call_args, field)

}  // namespace ov::intel_cpu
