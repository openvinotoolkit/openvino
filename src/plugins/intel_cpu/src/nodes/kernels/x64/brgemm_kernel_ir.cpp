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

// Emitting an M tile unrolls the whole accumulator block, so a shape with
// many tiles would produce an enormous kernel. oneDNN loops over them; we
// do not have a reason to yet, and a cap is honest about it.
//
// @todo claude: loop over bdb instead of unrolling, once a shape that
// needs it is in the slice.
constexpr dim_t max_unrolled_m_blocks = 4;

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

    // The generator assumes one accumulator register per 16 columns.
    if (brg.ld_block != 16) {
        return "only a 16-column ld_block";
    }

    if (brg.bdb > max_unrolled_m_blocks) {
        return "too many M blocks to unroll";
    }

    // A degenerate descriptor would emit a kernel that stores nothing.
    if (brg.bd_block <= 0 || brg.ld_block2 <= 0 || brg.rdb <= 0 || brg.rd_block <= 0) {
        return "degenerate blocking";
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
    OPENVINO_ASSERT(unsupported_reason(m_brg) == nullptr,
                    "brgemm_kernel_ir::generate: descriptor outside the supported slice");

    // f32 on AVX-512: one accumulator register holds 16 columns, which is
    // what oneDNN calls ld_block.
    constexpr size_t N = 16;
    OPENVINO_ASSERT(static_cast<size_t>(m_brg.ld_block) == N,
                    "brgemm_kernel_ir: unexpected ld_block ", m_brg.ld_block);

    using Params = brgemm_kernel_params_t;
    using element = brgemm_batch_element_t;

    // The blocking is oneDNN's, taken from the descriptor rather than
    // re-derived: bd_block rows by ld_block2 column groups per tile, bdb
    // tiles down M, rdb steps of rd_block along K.
    const auto bd_block = static_cast<size_t>(m_brg.bd_block);
    const auto ld_block2 = static_cast<size_t>(m_brg.ld_block2);
    const auto bdb = static_cast<size_t>(m_brg.bdb);
    const auto rdb = static_cast<size_t>(m_brg.rdb);
    const auto rd_block = static_cast<size_t>(m_brg.rd_block);
    const auto lda = static_cast<size_t>(m_brg.LDA);
    const auto ldb = static_cast<size_t>(m_brg.LDB);
    const auto ldc = static_cast<size_t>(m_brg.LDC);
    constexpr size_t ts = sizeof(float);

    preamble();
    set_vec_width(N * ts * 8);  // 512: the whole register file is allocable
    begin_ir();

    auto batch_base = arg<const element*>(&Params::batch);
    auto c_base = arg<float*>(&Params::ptr_C);
    auto bs_count = arg(&Params::BS);

    // One M tile at a time, unrolled at record time because bdb is known
    // here and small (capped in unsupported_reason). Each tile owns its
    // accumulators and stores them before the next begins, so peak
    // pressure is one tile, not all of them.
    for (size_t m_blk = 0; m_blk < bdb; ++m_blk) {
        const size_t row0 = m_blk * bd_block;

        // The accumulator tile. Defined before the batch loop: an
        // accumulator initialized inside a loop restarts every trip.
        std::vector<variable<float[N]>> acc;
        acc.reserve(bd_block * ld_block2);
        for (size_t i = 0; i < bd_block * ld_block2; ++i) {
            acc.push_back(ir_zero<N>());
        }
        auto at = [&](size_t bd, size_t ld) -> variable<float[N]>& {
            return acc[bd * ld_block2 + ld];
        };

        // Walks the batch descriptor array; A and B come from it, one
        // pair per batch element (brgemm_addr).
        auto cursor = ir_def_gpr({batch_base.vid()}, gpr_copy(), "batch_cursor");
        auto cursor_var = variable<const element*>(*this, cursor);

        foreach(size_t{0}, bs_count, [&](const variable<size_t>&) {
            auto a_ptr = ir_load_gpr<const float*>(cursor_var, offsetof(element, ptr.A));
            auto b_ptr = ir_load_gpr<const float*>(cursor_var, offsetof(element, ptr.B));

            // This tile's rows start part-way down A.
            if (row0 != 0) {
                ir_advance(a_ptr, row0 * lda * ts);
            }

            foreach(size_t{0}, rdb, [&](const variable<size_t>&) {
                // rd_block reduction steps unrolled inside the loop body,
                // matching how oneDNN blocks K. Everything addressed off
                // the two cursors with constant displacements.
                for (size_t rd = 0; rd < rd_block; ++rd) {
                    std::vector<variable<float[N]>> b_col;
                    b_col.reserve(ld_block2);
                    for (size_t ld = 0; ld < ld_block2; ++ld) {
                        b_col.push_back(ir_load<N>(b_ptr, (rd * ldb + ld * N) * ts));
                    }
                    for (size_t bd = 0; bd < bd_block; ++bd) {
                        auto a_val = ir_broadcast<N>(a_ptr, (bd * lda + rd) * ts);
                        for (size_t ld = 0; ld < ld_block2; ++ld) {
                            ir_accumulate(at(bd, ld), Insn3::fmadd231ps, a_val, b_col[ld]);
                        }
                    }
                }
                ir_advance(a_ptr, rd_block * ts);
                ir_advance(b_ptr, rd_block * ldb * ts);
            });

            ir_advance(cursor_var, sizeof(element));
        });

        for (size_t bd = 0; bd < bd_block; ++bd) {
            for (size_t ld = 0; ld < ld_block2; ++ld) {
                ir_store<N>(c_base, ((row0 + bd) * ldc + ld * N) * ts, at(bd, ld));
            }
        }
    }

    end_ir();
    postamble();
}

}  // namespace ov::intel_cpu::kernel

#endif  // OPENVINO_ARCH_X86_64
