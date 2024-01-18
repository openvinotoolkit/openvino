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
struct jit_snippets_call_args {
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *buffer_scratchpad_ptr = nullptr;
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
    jit_kernel_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void init_data_pointers(const Xbyak::Reg64&, const Xbyak::Reg64&, const std::vector<Xbyak::Reg64>&) const;

    jit_snippets_compile_args jcp;
    std::vector<size_t> gp_regs_pool;
    std::vector<size_t> master_shape;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_unique_buffers;
    // Vector of indices (lenght = input tensor rank) per every input and output that describes in which order
    // corresponding tensor dimensions are accessed (default: consecutive dense, e.g. 0,1,2,3 for 4D tensor).
    // Needed to calc i/o offsets.
    std::vector<std::vector<size_t>> io_data_layouts;
    std::vector<std::vector<size_t>> io_shapes = {};
    std::vector<size_t> io_data_sizes {};

    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;

    const size_t reg_indexes_idx;
    const size_t reg_const_params_idx;
};

}   // namespace intel_cpu
}   // namespace ov
