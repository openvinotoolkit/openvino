// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
// #include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu {

struct ParallelLoopConfig : public ov::snippets::KernelExecutorBase::GenericConfig {
public:
    ParallelLoopConfig() = default;
    ParallelLoopConfig(jit_snippets_call_args::loop_args_t loop_args, int num_threads)
        : m_loop_args(std::move(loop_args)),
          m_num_threads(num_threads) {}

    bool is_completed() const override {
        // todo: do we need a more detailed check on whether the work_amount is initialized properly?
        return m_loop_args.m_work_amount >= 0 && m_num_threads > 0;
    }

    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<ParallelLoopConfig>(new ParallelLoopConfig(*this));
    }

    virtual size_t hash() const override;

    const jit_snippets_call_args::loop_args_t& get_loop_args() const {
        return m_loop_args;
    }
    int get_num_threads() const {
        return m_num_threads;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override {
        // TODO: implement a more detailed string representation
        return {};
    };
#endif

protected:
    jit_snippets_call_args::loop_args_t m_loop_args{};
    int m_num_threads{0};
};

// Note: the ParallelLoopKernel is empty because this executor doesn't need a kernel
class ParallelLoopKernel {};

class ParallelLoopExecutor : public snippets::KernelExecutor<ParallelLoopConfig, ParallelLoopKernel> {
public:
    ParallelLoopExecutor(ParallelLoopConfig config) : KernelExecutor(std::move(config)) {}
    typedef void (*loop_preamble_t)(int64_t, void*);
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
