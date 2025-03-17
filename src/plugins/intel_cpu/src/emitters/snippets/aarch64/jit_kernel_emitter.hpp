// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov::intel_cpu::aarch64 {

///
/// \brief    Kernel is the only entry point to Codogen Jit compilation. Kernel perform abstract-to-physical register
/// mapping and creates a pools of available gpr and vec registers. Kernel usually contains (at least one)
/// jit_loop_begin_emitter and jit_loop_end_emitter pair. In general the enclosed emitters should be organized in the
/// following way: jit_kernel_emitter {                 /* entry point, maps registers, creates pools of available
/// registers */
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

class jit_kernel_emitter : public jit_emitter {
public:
    jit_kernel_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                       dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 0;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    /**
     * @brief populates physical registers pools for x86 (both vec and gp).
     * Skips stack-related gprs and extra gprs passed as arguments.
     * @arg gpr_blacklist - set of gp registers that should not be added to register pool
     * @arg vec_blacklist - set of vec registers should not be added to register pool
     */
    void init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist);

    virtual void init_data_pointers(const std::vector<Xbyak_aarch64::XReg>& arg_regs,
                                    const std::vector<Xbyak_aarch64::XReg>& data_ptr_regs) const = 0;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    jit_snippets_compile_args jcp;
    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;
    std::vector<size_t> gp_regs_pool;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t num_unique_buffers = 0;

    std::shared_ptr<snippets::lowered::LinearIR> body;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_kernel_emitter(const jit_kernel_emitter* emitter);
#endif
};

class jit_kernel_static_emitter : public jit_kernel_emitter {
public:
    jit_kernel_static_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                              dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_count() const override {
        return 2;
    }

private:
    void init_data_pointers(const std::vector<Xbyak_aarch64::XReg>& arg_regs,
                            const std::vector<Xbyak_aarch64::XReg>& data_ptr_regs) const override;

    std::vector<size_t> master_shape;
    std::vector<std::vector<size_t>> data_offsets;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_kernel_static_emitter(const jit_kernel_static_emitter* emitter);
#endif
};

class jit_kernel_dynamic_emitter : public jit_kernel_emitter {
public:
    jit_kernel_dynamic_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_count() const override {
        return 1;
    }

private:
    void init_data_pointers(const std::vector<Xbyak_aarch64::XReg>& arg_regs,
                            const std::vector<Xbyak_aarch64::XReg>& data_ptr_regs) const override;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_kernel_dynamic_emitter(const jit_kernel_dynamic_emitter* emitter);
#endif
};

}  // namespace ov::intel_cpu::aarch64
