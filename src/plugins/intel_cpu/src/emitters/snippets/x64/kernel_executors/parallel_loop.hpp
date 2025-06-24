// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov::intel_cpu {

struct ParallelLoopConfig : public ov::snippets::KernelExecutorBase::GenericConfig {
public:
    ParallelLoopConfig() = default;
    ParallelLoopConfig(const jit_snippets_call_args::loop_args_t& loop_args, size_t increment)
        : m_loop_args(loop_args),
          m_increment(increment) {}

    [[nodiscard]] bool is_completed() const override {
        // todo: do we need a more detailed check on whether the work_amount is initialized properly?
        return m_loop_args.m_work_amount >= 0;
    }

    [[nodiscard]] std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::make_unique<ParallelLoopConfig>(*this);
    }

    [[nodiscard]] size_t hash() const override;

    [[nodiscard]] const jit_snippets_call_args::loop_args_t& get_loop_args() const {
        return m_loop_args;
    }
    [[nodiscard]] size_t get_increment() const {
        return m_increment;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    [[nodiscard]] std::string to_string() const override {
        // TODO: implement a more detailed string representation
        return {};
    };
#endif

protected:
    jit_snippets_call_args::loop_args_t m_loop_args;
    size_t m_increment = 0;
};

// Note: the ParallelLoopKernel is empty because this executor doesn't need a kernel
class ParallelLoopKernel {};

class ParallelLoopExecutor : public snippets::KernelExecutor<ParallelLoopConfig, ParallelLoopKernel> {
public:
    ParallelLoopExecutor(ParallelLoopConfig config) : KernelExecutor(std::move(config)) {}
    using loop_preamble_t = void (*)(int64_t, void*);
    /** Function that will be called in runtime to execute the kernel */
    static void execute(const ParallelLoopExecutor* executor, uintptr_t** stack_ptr, loop_preamble_t preamble_ptr);

protected:
    /*** Updates stored kernel config based on runtime info from expression (e.g. new input shapes). */
    void update_config(const snippets::lowered::ExpressionPtr& expr,
                       const snippets::lowered::LinearIRCPtr& linear_ir,
                       ParallelLoopConfig& config) const override {}
    /*** Updates stored kernel in accordance with the passed config. Recompilation of the kernel is
     * performed if necessary. */
    void update_kernel(const ParallelLoopConfig& c, std::shared_ptr<ParallelLoopKernel>& kernel) const override;
};

}  // namespace ov::intel_cpu
