// Copyright (C) 2020-2024 Intel Corporation
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

    bool is_completed() const override {
        return m_work_amount >= 0 && m_num_threads > 0;
    }

    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<ParallelLoopConfig>(new ParallelLoopConfig(*this));
    }

    virtual size_t hash() const override;

    int64_t get_work_amount() const { return m_work_amount; }
    int get_num_threads() const { return  m_num_threads; }

// todo: re-enable
//#ifdef SNIPPETS_DEBUG_CAPS
//    std::string to_string() const override;
//#endif

protected:

    int64_t m_work_amount{0};
    int m_num_threads {0};

};


class ParallelLoopExecutor : public snippets::KernelExecutor<ParallelLoopConfig, void> {
public:
    ParallelLoopExecutor(ParallelLoopConfig config);
    typedef void(*loop_preamble_t)(int64_t , int64_t , void*);
    /** Function that will be called in runtime to execute the kernel */
    static void execute(const ParallelLoopExecutor* executor, void* stack_ptr, loop_preamble_t preamble_ptr);

protected:
    /*** Updates stored kernel config based on runtime info from expression (e.g. new input shapes). */
    virtual void update_config(const lowered::ExpressionPtr& expr, const lowered::LinearIRCPtr& linear_ir, Conf& config) const = 0;
    /*** Updates stored kernel in accordance with the passed config. Recompilation of the kernel is
     * performed if necessary. */
    virtual void update_kernel(const Conf& c, std::shared_ptr<KernelType>& kernel) const = 0;
};


}  // namespace ov::intel_cpu
