// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_kernel_ir.hpp"

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#    include <cpu/x64/cpu_isa_traits.hpp>

#    include <cstdlib>
#    include <iostream>
#    include <string>

#    include "openvino/core/except.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::kernel {

namespace {

// Increment 3 turns this on. Until then the predicate below describes the
// intended slice but accepts nothing, so oneDNN keeps serving every
// descriptor and enabling OV_JIT_IR_BRGEMM changes no generated code.
//
// @todo claude: no code generator yet — generate() throws.
constexpr bool generator_available = false;

}  // namespace

brgemm_kernel_ir::brgemm_kernel_ir(const brgemm_desc_t& brg)
    : jit_kernel(jit_name()),
      m_brg(brg) {}

brgemm_kernel_ir::mode brgemm_kernel_ir::env_mode() {
    static const mode value = [] {
        const char* env = std::getenv("OV_JIT_IR_BRGEMM");
        if (env == nullptr) {
            return mode::off;
        }
        const std::string requested(env);
        if (requested == "1") {
            return mode::offer;
        }
        if (requested == "2") {
            return mode::force;
        }
        return mode::off;
    }();
    return value;
}

const char* brgemm_kernel_ir::unsupported_reason(const brgemm_desc_t& brg) {
    if (!generator_available) {
        return "no code generator yet";
    }

    // The first slice: a plain f32 batched GEMM and nothing else. Each
    // line here is a feature of the built-in kernel that has to be
    // reproduced and differentially tested before it can be removed.

    // Data types: one, and the same one throughout.
    if (brg.dt_a != data_type::f32 || brg.dt_b != data_type::f32 ||
        brg.dt_c != data_type::f32 || brg.dt_d != data_type::f32) {
        return "only f32 throughout";
    }

    // AVX-512 only. AVX2 needs the epilogue tail strategy and a 16-register
    // budget; both work in the DSL but change the accumulator blocking.
    if (!is_superset(brg.isa_impl, avx512_core)) {
        return "only avx512_core and above";
    }

    // AMX is a separate register file plus tile configuration held as
    // machine state — a fourth register class and an analogue of RVV's vl,
    // neither of which the IR models yet.
    if (brg.is_tmm || brg.is_dgmm) {
        return "AMX tiles and depthwise are not modelled";
    }

    // Batch kind: the pointer-array form BrgemmKernel (MHA/SDPA) uses.
    // brgemm_strd, which snippets uses, is the next one to add.
    if (brg.type != brgemm_addr) {
        return "only brgemm_addr batching";
    }

    // alpha != 1 and beta != 0 are a scale and a read-modify-write of C.
    if (brg.alpha != 1.0F || brg.beta != 0.0F) {
        return "only alpha 1 and beta 0";
    }

    // Post-ops run through jit_uni_postops_injector, which reserves and
    // preserves registers by its own rules inside a region the allocator
    // believes it owns. That needs the clobber-barrier op the IR does not
    // have yet.
    if (brg.with_binary || brg.with_sum || brg.with_eltwise || brg.with_bias ||
        brg.with_src_scales || brg.with_wei_scales || brg.with_dst_scales) {
        return "post-ops, bias and scales are not supported";
    }

    // Quantization, compensation and weight decompression: each is an
    // extra pass over the accumulators with its own reserved registers.
    if (brg.is_int8 || brg.req_s8s8_compensation || brg.req_cal_comp_pads ||
        brg.zp_type_a != brgemm_broadcast_t::none ||
        brg.zp_type_b != brgemm_broadcast_t::none ||
        brg.zp_type_c != brgemm_broadcast_t::none || brg.with_src_dyn_quant ||
        brg.with_wei_decomp) {
        return "quantization, compensation and weight decompression are not supported";
    }

    // Virtual padding makes the row range of each iteration dynamic.
    if (brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0) {
        return "virtual padding is not supported";
    }

    // Tails last: a full-width N keeps the first kernel to unmasked
    // stores. The ld tail then reuses the active-length machinery the DSL
    // already has, which is the cheapest of the widenings.
    if (brg.ldb_tail != 0 || brg.bdb_tail != 0 || brg.rdb_tail != 0) {
        return "tails are not supported";
    }

    return nullptr;
}

status_t brgemm_kernel_ir::factory(dnnl::impl::cpu::x64::brgemm_kernel_t** kernel,
                                   const brgemm_desc_t& brg) {
    if (const char* reason = unsupported_reason(brg)) {
        if (env_mode() != mode::force) {
            return status::unimplemented;  // oneDNN's generators take over
        }
        // Forced: refuse to let the fallback hide the gap, and say which
        // descriptor was declined so the message is actionable.
        std::cerr << "[brgemm_kernel_ir] OV_JIT_IR_BRGEMM=2 and this descriptor is not"
                     " supported: "
                  << reason << " (M=" << brg.bcast_dim << " N=" << brg.load_dim
                  << " K=" << brg.reduce_dim << " dt_a=" << static_cast<int>(brg.dt_a)
                  << " dt_b=" << static_cast<int>(brg.dt_b) << " beta=" << brg.beta
                  << ")\n";
        return status::runtime_error;
    }
    *kernel = new brgemm_kernel_ir(brg);  // NOLINT(cppcoreguidelines-owning-memory)
    return status::success;
}

void brgemm_kernel_ir::register_factory(mode m) {
    brgemm_kernel_set_factory(m == mode::off ? nullptr : &brgemm_kernel_ir::factory);
}

status_t brgemm_kernel_ir::create_kernel() {
    const status_t st = jit_generator_t::create_kernel();
    if (st != status::success) {
        return st;
    }
    m_fn = reinterpret_cast<function_t>(const_cast<uint8_t*>(jit_ker()));  // NOLINT
    return status::success;
}

void brgemm_kernel_ir::operator()(brgemm_kernel_params_t* params) const {
    m_fn(params);
}

void brgemm_kernel_ir::generate() {
    // @todo claude: increment 3. is_supported() accepts nothing yet, so
    // this is unreachable; it throws rather than emitting an empty kernel,
    // because a kernel that returns without writing C is a silent wrong
    // answer and that is the failure mode this branch has already paid
    // for once (the spill stub that emitted nothing).
    OPENVINO_THROW("brgemm_kernel_ir: no code generator yet");
}

}  // namespace ov::intel_cpu::kernel

#endif  // OPENVINO_ARCH_X86_64
