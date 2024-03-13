// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include "jit_container_emitter.hpp"


namespace ov {
namespace intel_cpu {

#define SNIPPETS_MAX_SNIPPETS_DIMS 12
#define SNIPPETS_MAX_HARNESS_DIMS 5
#define SNIPPETS_MAX_TILE_RANK 2
#define SNIPPETS_DYNAMIC_MASTER_SHAPE_RANK 6

#define GET_OFF(field) offsetof(jit_snippets_call_args, field)
#define GET_OFF_LOOP_ARGS(field) offsetof(jit_snippets_call_args::loop_args_t, field)

struct jit_snippets_call_args {
    struct loop_args_t;

    jit_snippets_call_args() = default;
    ~jit_snippets_call_args();

    void register_loops(const std::vector<loop_args_t>& loops);

    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *buffer_scratchpad_ptr = nullptr;

    // Note: Ideally loop_args must be private, since we manage this pointer manually.
    // However, standard-layout class definition (to use offset_of) requires the same access specifier
    // for all non-static data members. So we can keep them public or friend all control-flow emitters
    int32_t num_loops = 0;
    loop_args_t* loop_args = nullptr;
};

struct jit_snippets_call_args::loop_args_t {
    friend class jit_loop_begin_dynamic_emitter;
    friend class jit_loop_end_dynamic_emitter;

    loop_args_t() = default;
    loop_args_t(int64_t work_amount, const std::vector<int64_t>& ptr_increments, const std::vector<int64_t>& finalization_offsets);
    loop_args_t(const loop_args_t& other);
    ~loop_args_t();

    loop_args_t& operator=(loop_args_t other);
    friend void swap(loop_args_t& first, loop_args_t& second);

private:
    void init_pointers_and_copy_data(const int64_t num_elements, const int64_t* ptr_increments, const int64_t* finalization_offsets);

    int64_t m_work_amount = 0;
    int64_t m_num_data_ptrs = 0;
    int64_t* m_ptr_increments = nullptr;
    int64_t* m_finalization_offsets = nullptr;
};

struct jit_snippets_compile_args {
    size_t parallel_executor_ndims = 1;
};

///
/// \brief    Kernel is the only entry point to Codogen Jit compilation. Kernel perform abstract-to-physical register
/// mapping and creates a pools of available gpr and vec registers. Kernel usually contains (at least one)
/// jit_loop_begin_emitter and jit_loop_end_emitter pair. In general the enclosed emitters should be organized in the following way:
/// jit_kernel_emitter {                 /* entry point, maps registers, creates pools of available registers */
///     1.S jit_loop_begin_emitter        /* Scalar Loop over the outer dimension [START] */
///         2.S jit_loop_begin_emitter    /* inner vector loop [START] */
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         2.E jit_loop_end_emitter      /* inner vector loop [END] */
///         3.S jit_loop_begin_emitter    /* inner scalar loop for tail processing [START]*/
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         3.E jit_loop_end_emitter      /* inner scalar loop for tail processing [END]*/
///     1.E jit_loop_end_emitter          /* Scalar Loop over the outer dimension [END] */
/// }
/// Note that Kernel doesn't accept any input arguments.
///

class jit_kernel_emitter : public jit_container_emitter {
public:
    jit_kernel_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void init_body_regs(const std::set<size_t>& kernel_regs, const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {});
    /**
    * @brief populates physical registers pools for x86 (both vec and gp).
     * Skips stack-related gprs and extra gprs passed as arguments.
     * @arg gpr_blacklist - set of gp registers that should not be added to register pool
     * @arg vec_blacklist - set of vec registers should not be added to register pool
    */
    void init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist);

    virtual void init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const = 0;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    jit_snippets_compile_args jcp;
    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;
    std::vector<size_t> gp_regs_pool;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t num_unique_buffers = 0;

    snippets::lowered::LinearIR::container mem_access_exprs;
    snippets::lowered::LinearIR::container general_exprs;

    const size_t reg_runtime_params_idx{0};

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_kernel_emitter(const jit_kernel_emitter *emitter);
#endif
};

class jit_kernel_static_emitter : public jit_kernel_emitter {
public:
    jit_kernel_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr);

private:
    void init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const override;

    const size_t reg_indexes_idx{1};
    std::vector<size_t> master_shape;

    // Vector of indices (lenght = input tensor rank) per every input and output that describes in which order
    // corresponding tensor dimensions are accessed (default: consecutive dense, e.g. 0,1,2,3 for 4D tensor).
    // Needed to calc i/o offsets.
    std::vector<std::vector<size_t>> io_data_layouts = {};
    std::vector<std::vector<size_t>> io_shapes = {};
    std::vector<size_t> io_data_sizes {};

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_kernel_static_emitter(const jit_kernel_static_emitter *emitter);
#endif
};

class jit_kernel_dynamic_emitter : public jit_kernel_emitter {
public:
    jit_kernel_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr);

private:
    void init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const override;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_kernel_dynamic_emitter(const jit_kernel_dynamic_emitter *emitter);
#endif
};

}   // namespace intel_cpu
}   // namespace ov
