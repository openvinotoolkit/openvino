// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// A BRGEMM kernel generated through the jit_kernel IR mode, offered to
// oneDNN as an alternative to its own generators.
//
// oneDNN calls the registered factory before its own dispatch (see
// brgemm_kernel_set_factory). The factory answers status::unimplemented
// for every descriptor outside the supported slice, and oneDNN's
// generators take over — so coverage is declared by an answer rather than
// by an omission, and widening it is a change to one predicate.
//
// The descriptor arrives finalized. Blocking (bd_block, ld_block2, the
// register budget behind them) has already been chosen by
// brgemm_utils.cpp, and this kernel consumes it rather than choosing its
// own. That is what makes a comparison against the built-in kernel
// meaningful: same shape, same blocking, two code generators.

#pragma once

// Defines OPENVINO_ARCH_X86_64, so it has to precede the guard below.
#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#    include <cpu/x64/brgemm/brgemm.hpp>
#    include <cpu/x64/brgemm/brgemm_types.hpp>

#    include "jit_kernel.hpp"

namespace ov::intel_cpu::kernel {

struct brgemm_kernel_ir : public dnnl::impl::cpu::x64::brgemm_kernel_t, public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(brgemm_kernel_ir)

    using brgemm_desc_t = dnnl::impl::cpu::x64::brgemm_desc_t;
    using brgemm_kernel_params_t = dnnl::impl::cpu::x64::brgemm_kernel_params_t;
    using status_t = dnnl::impl::status_t;

    explicit brgemm_kernel_ir(const brgemm_desc_t& brg);

    // Why this generator will not take a descriptor, or nullptr when it
    // will. A reason rather than a bool because the answer has to be
    // reportable: in force mode it is the error message, and in the
    // differential test it is the skip message. A bare false tells you a
    // shape was declined but not which line declined it.
    static const char* unsupported_reason(const brgemm_desc_t& brg);

    // Which descriptors this generator handles. Everything else falls
    // through to oneDNN. Deliberately narrow; widened one flag at a time,
    // each widening paid for by the differential test.
    static bool is_supported(const brgemm_desc_t& brg) {
        return unsupported_reason(brg) == nullptr;
    }

    // Registered with oneDNN via brgemm_kernel_set_factory(). Returns a
    // constructed but not yet created kernel: oneDNN calls create_kernel()
    // itself, so a generation failure is reported through one path.
    static status_t factory(dnnl::impl::cpu::x64::brgemm_kernel_t** kernel,
                            const brgemm_desc_t& brg);

    // OV_JIT_IR_BRGEMM:
    //   unset/0  off    — the factory is not installed at all
    //   1        offer  — take what is supported, let oneDNN have the rest
    //   2        force  — take what is supported and fail on the rest
    //
    // `force` exists because `offer` cannot tell a passing test from a
    // test that never ran this generator: an unsupported descriptor falls
    // through to oneDNN and everything stays green. Under `force` the
    // same descriptor is a hard error naming the reason, so coverage gaps
    // surface instead of hiding. Not for production — a model containing
    // one unsupported shape will refuse to compile.
    enum class mode : std::uint8_t { off, offer, force };

    static mode env_mode();

    // Installs the factory, or removes it when the mode is `off`. Called
    // once during plugin construction rather than from a static
    // initializer, so ordering against oneDNN's own initialization is not
    // a question.
    static void register_factory(mode m);

    // ── brgemm_kernel_t ───────────────────────────────────────────────
    // jit_generator_t declares create_kernel() with the same signature, so
    // this single override satisfies both bases.
    status_t create_kernel() override;
    void operator()(brgemm_kernel_params_t* params) const override;
    [[nodiscard]] const dnnl::impl::cpu::x64::jit_generator_t* get_jit_generator() const override {
        // Our own generator, so ONEDNN_JIT_DUMP and oneDNN verbose work on
        // this kernel with no extra plumbing.
        return this;
    }
    [[nodiscard]] const brgemm_desc_t& get_brg() const override { return m_brg; }

private:
    void generate() override;

    using function_t = void (*)(const brgemm_kernel_params_t*);

    brgemm_desc_t m_brg;
    function_t m_fn = nullptr;
};

}  // namespace ov::intel_cpu::kernel

#endif  // OPENVINO_ARCH_X86_64
