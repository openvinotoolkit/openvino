// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gdn_jit_kernel.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cmath>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <memory>

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_gdn_call_args, field)

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::load(const Vmm& vmm_dst,
                               const Xbyak::Reg64& reg_src,
                               ov::element::Type src_prc,
                               const int& elt_num,
                               bool fill,
                               size_t offset,
                               ov::element::Type dst_prc) {
    // Typed load helper (src_prc -> dst_prc VMM via jit emitter)
    const auto seed = load_emitter_params(src_prc, dst_prc, elt_num, fill, "float_min").hash();
    if (!emitters[seed]) {
        constexpr cpu_isa_t load_isa = ((isa & zmm_bit) != 0) ? avx512_core : isa;
        emitters[seed] =
            std::make_unique<jit_load_emitter>(this, load_isa, src_prc, dst_prc, elt_num, dst_prc, fill, "float_min");
    }
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), offset},
                              {static_cast<size_t>(vmm_dst.getIdx())},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::store(const Xbyak::Reg64& reg_dst,
                                const Vmm& vmm_src,
                                ov::element::Type dst_prc,
                                const int& elt_num,
                                size_t offset,
                                ov::element::Type src_prc) {
    // Typed store helper (src_prc VMM -> dst_prc via jit emitter)
    const auto seed = store_emitter_params(src_prc, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        constexpr cpu_isa_t store_isa = ((isa & zmm_bit) != 0) ? avx512_core : isa;
        emitters[seed] = std::make_unique<jit_store_emitter>(this, store_isa, src_prc, dst_prc, elt_num);
    }
    emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx())},
                              {static_cast<size_t>(reg_dst.getIdx()), offset},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::reduce_zmm_f32_to_xmm_scalar(const Xbyak::Zmm& zmm_src,
                                                       const Xbyak::Xmm& xmm_dst,
                                                       const Xbyak::Xmm& xmm_tmp0,
                                                       const Xbyak::Xmm& xmm_tmp1) {
    // Horizontal reduce 16x f32 (ZMM) into scalar lane of xmm_dst.
    // Prefer shuffle/add pattern over vhaddps on AVX-512.
    vextractf32x8(Xbyak::Ymm(xmm_tmp1.getIdx()), zmm_src, 1);
    vaddps(Xbyak::Ymm(xmm_tmp0.getIdx()), Xbyak::Ymm(zmm_src.getIdx()), Xbyak::Ymm(xmm_tmp1.getIdx()));
    vextractf128(xmm_tmp1, Xbyak::Ymm(xmm_tmp0.getIdx()), 1);
    vaddps(xmm_tmp0, xmm_tmp0, xmm_tmp1);
    vpermilps(xmm_tmp1, xmm_tmp0, 0xB1);
    vaddps(xmm_tmp0, xmm_tmp0, xmm_tmp1);
    vpermilps(xmm_tmp1, xmm_tmp0, 0x4E);
    vaddps(xmm_tmp0, xmm_tmp0, xmm_tmp1);
    vaddss(xmm_dst, xmm_dst, xmm_tmp0);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::dot_product_scalar(const Xbyak::Xmm& xmm_dst,
                                             const Xbyak::Reg64& reg_a,
                                             const Xbyak::Reg64& reg_b,
                                             size_t tail_count,
                                             size_t base_off,
                                             size_t elem_size,
                                             const Xbyak::Xmm& xmm_tmp0,
                                             const Xbyak::Xmm& xmm_tmp1) {
    // Tail handling via load-emitter masks: process full vectors then one masked vector.
    if (tail_count == 0) {
        return;
    }

    const size_t step = vec_size * elem_size;
    const size_t vec_cnt = tail_count / vec_size;
    const size_t tail = tail_count % vec_size;

    uni_vpxor(v_aux0, v_aux0, v_aux0);
    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = base_off + i * step;
        load(v_aux1, reg_a, m_jcp.data_prc, static_cast<int>(vec_size), false, off);
        load(v_aux2, reg_b, m_jcp.data_prc, static_cast<int>(vec_size), false, off);
        vfmadd231ps(v_aux0, v_aux1, v_aux2);
    }

    if (tail > 0) {
        const size_t off = base_off + vec_cnt * step;
        load(v_aux1, reg_a, m_jcp.data_prc, static_cast<int>(tail), false, off);
        load(v_aux2, reg_b, m_jcp.data_prc, static_cast<int>(tail), false, off);
        vfmadd231ps(v_aux0, v_aux1, v_aux2);
    }

    if constexpr (std::is_same_v<Vmm, Xbyak::Ymm>) {
        vextractf128(xmm_tmp1, v_aux0, 1);
        vaddps(xmm_tmp0, Xbyak::Xmm(v_aux0.getIdx()), xmm_tmp1);
        vpermilps(xmm_tmp1, xmm_tmp0, 0xB1);
        vaddps(xmm_tmp0, xmm_tmp0, xmm_tmp1);
        vpermilps(xmm_tmp1, xmm_tmp0, 0x4E);
        vaddps(xmm_tmp0, xmm_tmp0, xmm_tmp1);
        vaddss(xmm_dst, xmm_dst, xmm_tmp0);
    } else {
        reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst, xmm_tmp0, xmm_tmp1);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::dot_product_to_scalar(const Xbyak::Xmm& xmm_dst,
                                                const Xbyak::Reg64& reg_a,
                                                const Xbyak::Reg64& reg_b) {
    // Dot product dispatcher (f32 vectorized, otherwise scalar) -> scalar xmm_dst
    uni_vpxor(xmm_dst, xmm_dst, xmm_dst);
    const size_t qk = m_jcp.qk_head_size;

    if (m_jcp.data_prc == ov::element::f32) {
        const size_t vec_cnt = qk / vec_size;
        const size_t tail = qk % vec_size;

        uni_vpxor(v_aux0, v_aux0, v_aux0);

        for (size_t i = 0; i < vec_cnt; i++) {
            const size_t off = i * vec_bytes;
            load(v_aux1, reg_a, ov::element::f32, static_cast<int>(vec_size), false, off);
            load(v_aux2, reg_b, ov::element::f32, static_cast<int>(vec_size), false, off);
            vfmadd231ps(v_aux0, v_aux1, v_aux2);
        }

        if constexpr (std::is_same_v<Vmm, Xbyak::Ymm>) {
            vextractf128(x_tmp0, v_aux0, 1);
            vaddps(Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()), x_tmp0);
            vpermilps(x_tmp0, Xbyak::Xmm(v_aux0.getIdx()), 0xB1);
            vaddps(Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()), x_tmp0);
            vpermilps(x_tmp0, Xbyak::Xmm(v_aux0.getIdx()), 0x4E);
            vaddps(Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()), x_tmp0);
            vaddss(xmm_dst, xmm_dst, Xbyak::Xmm(v_aux0.getIdx()));
        } else {
            reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst, x_tmp0, x_tmp1);
        }

        dot_product_scalar(xmm_dst, reg_a, reg_b, tail, vec_cnt * vec_bytes, sizeof(float), x_tmp0, x_tmp1);
    } else {
        dot_product_scalar(xmm_dst, reg_a, reg_b, qk, 0, m_jcp.data_prc.size(), x_tmp0, x_tmp1);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::multiply_scalar(const Xbyak::Reg64& reg_vec, const Xbyak::Xmm& xmm_scalar) {
    // In-place vector scale: reg_vec[i] *= xmm_scalar
    const auto elt_num = static_cast<int>(vec_size);
    const size_t elem_size = m_jcp.data_prc.size();
    const size_t step = static_cast<size_t>(elt_num) * elem_size;
    const size_t vec_cnt = m_jcp.qk_head_size / static_cast<size_t>(elt_num);
    const size_t tail = m_jcp.qk_head_size % static_cast<size_t>(elt_num);

    vbroadcastss(v_tmp1, xmm_scalar);

    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = i * step;
        load(v_tmp0, reg_vec, m_jcp.data_prc, elt_num, false, off);
        vmulps(v_tmp0, v_tmp0, v_tmp1);
        store(reg_vec, v_tmp0, m_jcp.data_prc, elt_num, off);
    }
    if (tail > 0) {
        const size_t off = vec_cnt * step;
        load(v_tmp0, reg_vec, m_jcp.data_prc, static_cast<int>(tail), false, off);
        vmulps(v_tmp0, v_tmp0, v_tmp1);
        store(reg_vec, v_tmp0, m_jcp.data_prc, static_cast<int>(tail), off);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::l2norm_inplace(const Xbyak::Reg64& reg_vec,
                                         const Xbyak::Xmm& xmm_eps,
                                         const Xbyak::Xmm& xmm_tmp0,
                                         const Xbyak::Xmm& xmm_tmp1,
                                         const Xbyak::Xmm& xmm_sum) {
    // In-place L2 normalization over one head vector (with eps)
    const size_t qk = m_jcp.qk_head_size;
    const size_t vec_cnt = qk / vec_size;
    const size_t tail = qk % vec_size;

    uni_vpxor(xmm_sum, xmm_sum, xmm_sum);
    uni_vpxor(v_aux0, v_aux0, v_aux0);

    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = i * vec_bytes;
        load(v_aux1, reg_vec, ov::element::f32, static_cast<int>(vec_size), false, off);
        vfmadd231ps(v_aux0, v_aux1, v_aux1);
    }

    if (tail > 0) {
        const size_t off = vec_cnt * vec_bytes;
        load(v_aux1, reg_vec, ov::element::f32, static_cast<int>(tail), false, off);
        vfmadd231ps(v_aux0, v_aux1, v_aux1);
    }

    if constexpr (std::is_same_v<Vmm, Xbyak::Ymm>) {
        vextractf128(x_tmp1, v_aux0, 1);
        vaddps(xmm_sum, Xbyak::Xmm(v_aux0.getIdx()), x_tmp1);
        vpermilps(x_tmp1, xmm_sum, 0xB1);
        vaddps(xmm_sum, xmm_sum, x_tmp1);
        vpermilps(x_tmp1, xmm_sum, 0x4E);
        vaddps(xmm_sum, xmm_sum, x_tmp1);
    } else {
        reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_sum, xmm_tmp0, xmm_tmp1);
    }

    vaddss(xmm_sum, xmm_sum, xmm_eps);
    vsqrtss(xmm_sum, xmm_sum, xmm_sum);
    vmovd(xmm_tmp1, reg_one.cvt32());
    vdivss(xmm_tmp1, xmm_tmp1, xmm_sum);

    vbroadcastss(v_tmp1, xmm_tmp1);

    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = i * vec_bytes;
        load(v_tmp0, reg_vec, ov::element::f32, static_cast<int>(vec_size), false, off);
        vmulps(v_tmp0, v_tmp0, v_tmp1);
        store(reg_vec, v_tmp0, ov::element::f32, static_cast<int>(vec_size), off);
    }

    if (tail > 0) {
        const size_t off = vec_cnt * vec_bytes;
        load(v_tmp0, reg_vec, ov::element::f32, static_cast<int>(tail), false, off);
        vmulps(v_tmp0, v_tmp0, v_tmp1);
        store(reg_vec, v_tmp0, ov::element::f32, static_cast<int>(tail), off);
    }
}

// ============================================
// Native xf16 helpers
// ============================================

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::load_vector_native_xf16(Vmm* vmm_array, const Xbyak::Reg64& reg_src, int num_regs) {
    // Load fp16 vector (up to 4 ZMMs) directly from memory
    for (int i = 0; i < num_regs; i++) {
        vmovups(vmm_array[i], ptr[reg_src + i * 64]);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::store_vector_native_xf16(const Xbyak::Reg64& reg_dst, Vmm* vmm_array, int num_regs) {
    // Store fp16 vector (up to 4 ZMMs) to memory
    for (int i = 0; i < num_regs; i++) {
        vmovups(ptr[reg_dst + i * 64], vmm_array[i]);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::dot_product_native_xf16(const Xbyak::Xmm& xmm_dst, Vmm* vmm_a, Vmm* vmm_b, int num_regs) {
    if (m_jcp.data_prc == ov::element::bf16) {
        // bf16 path: accumulate directly in fp32 with vdpbf16ps
        uni_vpxor(v_aux0, v_aux0, v_aux0);
        for (int i = 0; i < num_regs; i++) {
            vdpbf16ps(v_aux0, vmm_a[i], vmm_b[i]);
        }
        uni_vpxor(xmm_dst, xmm_dst, xmm_dst);
        reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst, x_tmp0, x_tmp1);
        return;
    }

    // f16 path: native fp16 accumulation then fp32 reduction
    uni_vpxor(v_tmp0, v_tmp0, v_tmp0);  // fp16 accumulator (32 lanes)

    for (int i = 0; i < num_regs; i++) {
        vfmadd231ph(v_tmp0, vmm_a[i], vmm_b[i]);
    }

    vcvtph2ps(v_aux0, Xbyak::Ymm(v_tmp0.getIdx()));
    vextractf32x8(Xbyak::Ymm(x_tmp0.getIdx()), Xbyak::Zmm(v_tmp0.getIdx()), 1);
    vcvtph2ps(v_aux1, Xbyak::Ymm(x_tmp0.getIdx()));
    vaddps(v_aux0, v_aux0, v_aux1);

    uni_vpxor(xmm_dst, xmm_dst, xmm_dst);
    reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst, x_tmp0, x_tmp1);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::scale_vector_native_xf16(Vmm* vmm_array, const Xbyak::Xmm& xmm_scalar, int num_regs) {
    if (m_jcp.data_prc == ov::element::bf16) {
        // bf16 path: unpack->fp32 mul->pack bf16 per half
        vbroadcastss(v_aux2, xmm_scalar);

        for (int i = 0; i < num_regs; i++) {
            // lower 16 bf16 -> fp32
            vpmovzxwd(v_aux0, Xbyak::Ymm(vmm_array[i].getIdx()));
            vpslld(v_aux0, v_aux0, 16);
            vmulps(v_aux0, v_aux0, v_aux2);
            vcvtneps2bf16(Xbyak::Ymm(x_tmp0.getIdx()), v_aux0);

            // upper 16 bf16 -> fp32
            vextractf32x8(Xbyak::Ymm(v_aux1.getIdx()), Xbyak::Zmm(vmm_array[i].getIdx()), 1);
            vpmovzxwd(v_aux1, Xbyak::Ymm(v_aux1.getIdx()));
            vpslld(v_aux1, v_aux1, 16);
            vmulps(v_aux1, v_aux1, v_aux2);
            vcvtneps2bf16(Xbyak::Ymm(x_tmp1.getIdx()), v_aux1);

            vinsertf32x8(Xbyak::Zmm(vmm_array[i].getIdx()),
                         Xbyak::Zmm(x_tmp0.getIdx()),
                         Xbyak::Ymm(x_tmp1.getIdx()),
                         1);
        }
        return;
    }

    // f16 path
    vcvtps2ph(x_tmp0, xmm_scalar, 0);
    vpbroadcastw(v_aux2, x_tmp0);

    for (int i = 0; i < num_regs; i++) {
        vmulph(vmm_array[i], vmm_array[i], v_aux2);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::fmadd_vector_native_xf16(Vmm* vmm_dst,
                                                   Vmm* vmm_src,
                                                   const Xbyak::Xmm& xmm_scalar,
                                                   int num_regs) {
    if (m_jcp.data_prc == ov::element::bf16) {
        // bf16 path: unpack dst/src -> fp32 fma -> pack bf16 per half
        vbroadcastss(v_aux2, xmm_scalar);

        for (int i = 0; i < num_regs; i++) {
            // lower half
            vpmovzxwd(v_aux0, Xbyak::Ymm(vmm_dst[i].getIdx()));
            vpslld(v_aux0, v_aux0, 16);
            vpmovzxwd(v_aux1, Xbyak::Ymm(vmm_src[i].getIdx()));
            vpslld(v_aux1, v_aux1, 16);
            vfmadd231ps(v_aux0, v_aux1, v_aux2);
            vcvtneps2bf16(Xbyak::Ymm(x_tmp0.getIdx()), v_aux0);

            // upper half
            vextractf32x8(Xbyak::Ymm(v_aux0.getIdx()), Xbyak::Zmm(vmm_dst[i].getIdx()), 1);
            vpmovzxwd(v_aux0, Xbyak::Ymm(v_aux0.getIdx()));
            vpslld(v_aux0, v_aux0, 16);
            vextractf32x8(Xbyak::Ymm(v_aux1.getIdx()), Xbyak::Zmm(vmm_src[i].getIdx()), 1);
            vpmovzxwd(v_aux1, Xbyak::Ymm(v_aux1.getIdx()));
            vpslld(v_aux1, v_aux1, 16);
            vfmadd231ps(v_aux0, v_aux1, v_aux2);
            vcvtneps2bf16(Xbyak::Ymm(x_tmp1.getIdx()), v_aux0);

            vinsertf32x8(Xbyak::Zmm(vmm_dst[i].getIdx()), Xbyak::Zmm(x_tmp0.getIdx()), Xbyak::Ymm(x_tmp1.getIdx()), 1);
        }
        return;
    }

    // f16 path
    vcvtps2ph(x_tmp0, xmm_scalar, 0);
    vpbroadcastw(v_aux2, x_tmp0);

    for (int i = 0; i < num_regs; i++) {
        vfmadd231ph(vmm_dst[i], vmm_src[i], v_aux2);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::l2norm_inplace_native_xf16(Vmm* vmm_array, const Xbyak::Xmm& xmm_eps, int num_regs) {
    // L2 normalization: vmm /= sqrt(sum(vmm^2) + eps)
    uni_vpxor(v_aux0, v_aux0, v_aux0);  // fp32 accumulator

    if (m_jcp.data_prc == ov::element::bf16) {
        for (int i = 0; i < num_regs; i++) {
            vdpbf16ps(v_aux0, vmm_array[i], vmm_array[i]);
        }
    } else {
        for (int i = 0; i < num_regs; i++) {
            // lower 16 fp16 lanes
            vcvtph2ps(v_aux1, Xbyak::Ymm(vmm_array[i].getIdx()));
            vfmadd231ps(v_aux0, v_aux1, v_aux1);

            // upper 16 fp16 lanes
            vextractf32x8(Xbyak::Ymm(v_tmp0.getIdx()), Xbyak::Zmm(vmm_array[i].getIdx()), 1);
            vcvtph2ps(v_aux1, Xbyak::Ymm(v_tmp0.getIdx()));
            vfmadd231ps(v_aux0, v_aux1, v_aux1);
        }
    }

    // Reduce to scalar: sqrt(sum + eps), then compute reciprocal
    uni_vpxor(x_hk, x_hk, x_hk);
    reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), x_hk, x_tmp0, x_tmp1);
    vaddss(x_hk, x_hk, xmm_eps);
    vsqrtss(x_hk, x_hk, x_hk);

    vmovd(x_tmp1, reg_one.cvt32());
    vdivss(x_tmp1, x_tmp1, x_hk);  // reciprocal

    // Scale vector by reciprocal
    scale_vector_native_xf16(vmm_array, x_tmp1, num_regs);
}

// ============================================
// Buffer-based L2 norm helper for qk_head_size > 128
// ============================================

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::l2norm_buffer_compute_scale_native_xf16(const Xbyak::Reg64& reg_buffer,
                                                                  const Xbyak::Xmm& xmm_eps,
                                                                  const Xbyak::Xmm& xmm_scale_out,
                                                                  int num_regs,
                                                                  int num_chunks) {
    // Compute L2 norm scale: 1/sqrt(sum(x^2) + eps)
    // Accumulates across all chunks from buffer, returns scale factor
    uni_vpxor(v_aux0, v_aux0, v_aux0);
    if (m_jcp.data_prc == ov::element::f16) {
        uni_vpxor(v_tmp0, v_tmp0, v_tmp0);
    }

    // Accumulate sum of squares across all chunks
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * MAX_REGS_PER_VEC;
        const int chunk_regs = std::min(MAX_REGS_PER_VEC, num_regs - chunk_start);
        const size_t chunk_offset = chunk_start * XF16_ELEMS_PER_ZMM * m_jcp.data_prc.size();

        mov(reg_aux2, reg_buffer);
        add(reg_aux2, chunk_offset);
        load_vector_native_xf16(const_cast<Vmm*>(v_k), reg_aux2, chunk_regs);

        if (m_jcp.data_prc == ov::element::bf16) {
            for (int i = 0; i < chunk_regs; i++) {
                vdpbf16ps(v_aux0, v_k[i], v_k[i]);
            }
        } else {
            // fp16 path - accumulate in native fp16
            for (int i = 0; i < chunk_regs; i++) {
                vfmadd231ph(v_tmp0, v_k[i], v_k[i]);
            }
        }
    }

    // Convert fp16 to fp32 after all chunks
    if (m_jcp.data_prc == ov::element::f16) {
        vcvtph2ps(v_aux1, Xbyak::Ymm(v_tmp0.getIdx()));
        vextractf32x8(Xbyak::Ymm(x_tmp0.getIdx()), Xbyak::Zmm(v_tmp0.getIdx()), 1);
        vcvtph2ps(v_aux2, Xbyak::Ymm(x_tmp0.getIdx()));
        vaddps(v_aux0, v_aux1, v_aux2);
    }

    // Compute reciprocal: 1/sqrt(sum + eps)
    uni_vpxor(xmm_scale_out, xmm_scale_out, xmm_scale_out);
    reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_scale_out, x_tmp0, x_tmp1);
    vaddss(xmm_scale_out, xmm_scale_out, xmm_eps);
    vsqrtss(xmm_scale_out, xmm_scale_out, xmm_scale_out);
    vmovd(x_value, reg_one.cvt32());
    vdivss(xmm_scale_out, x_value, xmm_scale_out);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::scale_buffer_native_xf16(const Xbyak::Reg64& reg_buffer,
                                                   const Xbyak::Xmm& xmm_scale,
                                                   Vmm* vmm_temp,
                                                   int num_regs,
                                                   int num_chunks) {
    // Scale all chunks of a buffer by a scalar
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * MAX_REGS_PER_VEC;
        const int chunk_regs = std::min(MAX_REGS_PER_VEC, num_regs - chunk_start);
        const size_t chunk_offset = chunk_start * XF16_ELEMS_PER_ZMM * m_jcp.data_prc.size();

        mov(reg_aux2, reg_buffer);
        add(reg_aux2, chunk_offset);
        load_vector_native_xf16(vmm_temp, reg_aux2, chunk_regs);
        scale_vector_native_xf16(vmm_temp, xmm_scale, chunk_regs);
        store_vector_native_xf16(reg_aux2, vmm_temp, chunk_regs);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::load_qk(bool is_f32, bool use_registers, int num_regs, int num_chunks) {
    if (!is_f32 && use_registers) {
        // Load K, Q directly into registers
        load_vector_native_xf16(const_cast<Vmm*>(v_k), reg_key_seq, num_regs);
        load_vector_native_xf16(const_cast<Vmm*>(v_q), reg_query_seq, num_regs);

        // Optional L2 normalization
        if (m_jcp.fuse_qk_l2norm) {
            l2norm_inplace_native_xf16(const_cast<Vmm*>(v_k), x_eps_k, num_regs);
            l2norm_inplace_native_xf16(const_cast<Vmm*>(v_q), x_eps_q, num_regs);
        }

        // Scale query
        scale_vector_native_xf16(const_cast<Vmm*>(v_q), x_qscale, num_regs);
    } else {
        // Temp-buffer path: f32 and large-head xf16 share this skeleton.
        mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
        mov(reg_query_tmp, ptr[reg_args + GET_OFF(query_tmp)]);

        const auto copy_elt_num = static_cast<int>(vec_size);
        const size_t copy_step = static_cast<size_t>(copy_elt_num) * m_jcp.data_prc.size();
        const size_t vec_cnt = m_jcp.qk_head_size / static_cast<size_t>(copy_elt_num);
        const size_t tail = m_jcp.qk_head_size % static_cast<size_t>(copy_elt_num);

        // State/temp copy: keep storage precision (no conversion).
        for (size_t i = 0; i < vec_cnt; i++) {
            const size_t off = i * copy_step;
            load(v_tmp0, reg_key_seq, m_jcp.data_prc, copy_elt_num, false, off, m_jcp.data_prc);
            store(reg_key_tmp, v_tmp0, m_jcp.data_prc, copy_elt_num, off, m_jcp.data_prc);
            load(v_tmp1, reg_query_seq, m_jcp.data_prc, copy_elt_num, false, off, m_jcp.data_prc);
            store(reg_query_tmp, v_tmp1, m_jcp.data_prc, copy_elt_num, off, m_jcp.data_prc);
        }
        if (tail > 0) {
            const size_t off = vec_cnt * copy_step;
            load(v_tmp0, reg_key_seq, m_jcp.data_prc, static_cast<int>(tail), false, off, m_jcp.data_prc);
            store(reg_key_tmp, v_tmp0, m_jcp.data_prc, static_cast<int>(tail), off, m_jcp.data_prc);
            load(v_tmp1, reg_query_seq, m_jcp.data_prc, static_cast<int>(tail), false, off, m_jcp.data_prc);
            store(reg_query_tmp, v_tmp1, m_jcp.data_prc, static_cast<int>(tail), off, m_jcp.data_prc);
        }

        if (is_f32) {
            mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
            mov(reg_query_tmp, ptr[reg_args + GET_OFF(query_tmp)]);

            if (m_jcp.fuse_qk_l2norm) {
                l2norm_inplace(reg_key_tmp, x_eps_k, x_tmp0, x_tmp1, x_hk);
                l2norm_inplace(reg_query_tmp, x_eps_q, x_tmp0, x_tmp1, x_hk);
            }

            multiply_scalar(reg_query_tmp, x_qscale);
        } else {
            // Optional L2 normalization (process in chunks)
            if (m_jcp.fuse_qk_l2norm) {
                // Normalize K
                l2norm_buffer_compute_scale_native_xf16(reg_key_tmp, x_eps_k, x_beta, num_regs, num_chunks);
                scale_buffer_native_xf16(reg_key_tmp, x_beta, const_cast<Vmm*>(v_k), num_regs, num_chunks);

                // Normalize Q and combine with q_scale
                l2norm_buffer_compute_scale_native_xf16(reg_query_tmp, x_eps_q, x_beta, num_regs, num_chunks);
                vmulss(x_beta, x_beta, x_qscale);  // Combine: l2norm_scale * q_scale
                scale_buffer_native_xf16(reg_query_tmp, x_beta, const_cast<Vmm*>(v_q), num_regs, num_chunks);
            } else {
                // No L2 norm, just scale Q by q_scale
                scale_buffer_native_xf16(reg_query_tmp, x_qscale, const_cast<Vmm*>(v_q), num_regs, num_chunks);
            }
        }
    }
}

// ============================================
// Main native xf16 kernel
// ============================================

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::generate() {
    // JIT codegen for native xf16 path
    // For qk_head_size <= 128: register-resident Q/K/H
    // For qk_head_size > 128: use temp buffers

    auto exp_injector = std::make_shared<jit_uni_eltwise_injector_t<isa>>(this,
                                                                          dnnl::impl::alg_kind::eltwise_exp,
                                                                          0.F,
                                                                          0.F,
                                                                          1.F,
                                                                          dnnl::impl::data_type::f32,
                                                                          true,
                                                                          Xbyak::Reg64(Xbyak::Operand::RCX),
                                                                          Xbyak::Opmask(1),
                                                                          true,
                                                                          false,
                                                                          false,
                                                                          false);

    this->preamble();

    Xbyak::Label l_t_loop;
    Xbyak::Label l_end;

    mov(reg_args, abi_param1);

    const bool is_f32 = (m_jcp.data_prc == ov::element::f32);
    const size_t qk = m_jcp.qk_head_size;
    const auto data_size = static_cast<int>(m_jcp.data_prc.size());
    const bool use_registers = !is_f32 && (qk <= 128);
    const auto num_regs = is_f32 ? 0 : static_cast<int>(qk / XF16_ELEMS_PER_ZMM);
    const auto num_chunks = is_f32 ? 0 : (num_regs + MAX_REGS_PER_VEC - 1) / MAX_REGS_PER_VEC;

    // One-time setup
    exp_injector->load_table_addr();
    mov(reg_aux.cvt32(), float2int(m_jcp.k_l2_norm_eps));
    vmovd(x_eps_k, reg_aux.cvt32());
    mov(reg_aux.cvt32(), float2int(m_jcp.q_l2_norm_eps));
    vmovd(x_eps_q, reg_aux.cvt32());
    mov(reg_aux.cvt32(), float2int(m_jcp.q_scale));
    vmovd(x_qscale, reg_aux.cvt32());
    mov(reg_one.cvt32(), float2int(1.0F));

    mov(reg_key_seq, ptr[reg_args + GET_OFF(key_seq)]);
    mov(reg_query_seq, ptr[reg_args + GET_OFF(query_seq)]);
    mov(reg_gate_seq, ptr[reg_args + GET_OFF(gate_seq)]);
    mov(reg_beta_seq, ptr[reg_args + GET_OFF(beta_seq)]);
    mov(reg_value_seq, ptr[reg_args + GET_OFF(value_seq)]);
    mov(reg_out_seq, ptr[reg_args + GET_OFF(output_seq)]);
    mov(reg_t, ptr[reg_args + GET_OFF(t_size)]);

    if (!is_f32 && !use_registers) {
        mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
        mov(reg_query_tmp, ptr[reg_args + GET_OFF(query_tmp)]);
    }

    test(reg_t, reg_t);
    jz(l_end, T_NEAR);

    L(l_t_loop);
    {
        load_qk(is_f32, use_registers, num_regs, num_chunks);

        // Compute gate and beta once per timestep and share across V lanes
        load(Vmm(x_gate.getIdx()), reg_gate_seq, m_jcp.data_prc, 1, false);
        exp_injector->compute_vector_range(x_gate.getIdx(), x_gate.getIdx() + 1);
        load(Vmm(x_beta.getIdx()), reg_beta_seq, m_jcp.data_prc, 1, false);

        // accumulate dot product of two vectors into v_aux0
        auto accumulate_dot_product = [&](Vmm* vmm_a, Vmm* vmm_b, int chunk_regs) {
            if (m_jcp.data_prc == ov::element::bf16) {
                for (int i = 0; i < chunk_regs; i++) {
                    vdpbf16ps(v_aux0, vmm_a[i], vmm_b[i]);
                }
            } else {
                for (int i = 0; i < chunk_regs; i++) {
                    // lower 16 elements
                    vcvtph2ps(v_aux1, Xbyak::Ymm(vmm_a[i].getIdx()));
                    vcvtph2ps(v_aux2, Xbyak::Ymm(vmm_b[i].getIdx()));
                    vfmadd231ps(v_aux0, v_aux1, v_aux2);
                    // upper 16 elements
                    vextractf32x8(Xbyak::Ymm(v_aux1.getIdx()), Xbyak::Zmm(vmm_a[i].getIdx()), 1);
                    vcvtph2ps(v_aux1, Xbyak::Ymm(v_aux1.getIdx()));
                    vextractf32x8(Xbyak::Ymm(v_aux2.getIdx()), Xbyak::Zmm(vmm_b[i].getIdx()), 1);
                    vcvtph2ps(v_aux2, Xbyak::Ymm(v_aux2.getIdx()));
                    vfmadd231ps(v_aux0, v_aux1, v_aux2);
                }
            }
        };

        // Unrolled V lanes: all lanes share same K/Q path above
        for (size_t v_idx = 0; v_idx < m_jcp.v_tile; ++v_idx) {
            mov(reg_state, ptr[reg_args + GET_OFF(state)]);
            add(reg_state, static_cast<size_t>(v_idx * m_jcp.qk_head_size * m_jcp.data_prc.size()));

            // Preload value scalar to overlap memory latency with hk reduction work.
            mov(reg_aux2, reg_value_seq);
            add(reg_aux2, static_cast<size_t>(v_idx * m_jcp.data_prc.size()));
            load(Vmm(x_value.getIdx()), reg_aux2, m_jcp.data_prc, 1, false);

            if (!is_f32 && use_registers) {
                // Load H for current V lane
                load_vector_native_xf16(const_cast<Vmm*>(v_h), reg_state, num_regs);

                // Scale hidden state by exp(gate)
                scale_vector_native_xf16(const_cast<Vmm*>(v_h), x_gate, num_regs);

                // Compute hk = dot(H, K)
                dot_product_native_xf16(x_hk, const_cast<Vmm*>(v_h), const_cast<Vmm*>(v_k), num_regs);

                // delta = (value - hk) * beta
                vsubss(x_delta, x_value, x_hk);
                vmulss(x_delta, x_delta, x_beta);

                // Update: H += K * delta
                fmadd_vector_native_xf16(const_cast<Vmm*>(v_h), const_cast<Vmm*>(v_k), x_delta, num_regs);

                // Output: out = dot(H, Q)
                dot_product_native_xf16(x_out, const_cast<Vmm*>(v_h), const_cast<Vmm*>(v_q), num_regs);

                // Store output and H state for current V lane
                mov(reg_aux2, reg_out_seq);
                add(reg_aux2, static_cast<size_t>(v_idx * m_jcp.data_prc.size()));
                store(reg_aux2, Vmm(x_out.getIdx()), m_jcp.data_prc, 1);
                store_vector_native_xf16(reg_state, const_cast<Vmm*>(v_h), num_regs);
            } else {
                // Temp-buffer compute path shared by f32 and xf16-large.
                if (is_f32) {
                    multiply_scalar(reg_state, x_gate);

                    mov(reg_query_tmp, reg_state);
                    mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
                    dot_product_to_scalar(x_hk, reg_query_tmp, reg_key_tmp);
                } else {
                    // Scale H by exp(gate)
                    scale_buffer_native_xf16(reg_state, x_gate, const_cast<Vmm*>(v_h), num_regs, num_chunks);

                    // Compute hk = dot(H, K)
                    uni_vpxor(v_aux0, v_aux0, v_aux0);
                    for (int chunk = 0; chunk < num_chunks; chunk++) {
                        const int chunk_start = chunk * MAX_REGS_PER_VEC;
                        const int chunk_regs = std::min(MAX_REGS_PER_VEC, num_regs - chunk_start);
                        const size_t chunk_offset = chunk_start * XF16_ELEMS_PER_ZMM * m_jcp.data_prc.size();

                        mov(reg_aux2, reg_state);
                        add(reg_aux2, chunk_offset);
                        load_vector_native_xf16(const_cast<Vmm*>(v_h), reg_aux2, chunk_regs);

                        mov(reg_aux2, reg_key_tmp);
                        add(reg_aux2, chunk_offset);
                        load_vector_native_xf16(const_cast<Vmm*>(v_k), reg_aux2, chunk_regs);

                        accumulate_dot_product(const_cast<Vmm*>(v_h), const_cast<Vmm*>(v_k), chunk_regs);
                    }
                    uni_vpxor(x_hk, x_hk, x_hk);
                    reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), x_hk, x_tmp0, x_tmp1);
                }

                // delta = (value - hk) * beta
                vsubss(x_delta, x_value, x_hk);
                vmulss(x_delta, x_delta, x_beta);

                if (is_f32) {
                    // Update: H += K * delta
                    const auto update_elt_num = static_cast<int>(vec_size);
                    const size_t update_step = static_cast<size_t>(update_elt_num) * sizeof(float);
                    const size_t update_vec_cnt = m_jcp.qk_head_size / static_cast<size_t>(update_elt_num);
                    const size_t update_tail = m_jcp.qk_head_size % static_cast<size_t>(update_elt_num);

                    vbroadcastss(v_aux2, x_delta);

                    mov(reg_aux2, ptr[reg_args + GET_OFF(key_tmp)]);
                    for (size_t i = 0; i < update_vec_cnt; i++) {
                        const size_t off = i * update_step;
                        load(v_tmp0, reg_state, ov::element::f32, update_elt_num, false, off);
                        load(v_tmp1, reg_aux2, ov::element::f32, update_elt_num, false, off);
                        vfmadd231ps(v_tmp0, v_tmp1, v_aux2);
                        store(reg_state, v_tmp0, ov::element::f32, update_elt_num, off);
                    }
                    if (update_tail > 0) {
                        const size_t off = update_vec_cnt * update_step;
                        load(v_tmp0, reg_state, ov::element::f32, static_cast<int>(update_tail), false, off);
                        load(v_tmp1, reg_aux2, ov::element::f32, static_cast<int>(update_tail), false, off);
                        vfmadd231ps(v_tmp0, v_tmp1, v_aux2);
                        store(reg_state, v_tmp0, ov::element::f32, static_cast<int>(update_tail), off);
                    }

                    mov(reg_query_tmp, reg_state);
                    mov(reg_key_tmp, ptr[reg_args + GET_OFF(query_tmp)]);
                    dot_product_to_scalar(x_out, reg_query_tmp, reg_key_tmp);
                } else {
                    // Update: H += K * delta
                    for (int chunk = 0; chunk < num_chunks; chunk++) {
                        const int chunk_start = chunk * MAX_REGS_PER_VEC;
                        const int chunk_regs = std::min(MAX_REGS_PER_VEC, num_regs - chunk_start);
                        const size_t chunk_offset = chunk_start * XF16_ELEMS_PER_ZMM * m_jcp.data_prc.size();

                        mov(reg_aux2, reg_state);
                        add(reg_aux2, chunk_offset);
                        load_vector_native_xf16(const_cast<Vmm*>(v_h), reg_aux2, chunk_regs);

                        mov(reg_aux2, reg_key_tmp);
                        add(reg_aux2, chunk_offset);
                        load_vector_native_xf16(const_cast<Vmm*>(v_k), reg_aux2, chunk_regs);

                        fmadd_vector_native_xf16(const_cast<Vmm*>(v_h), const_cast<Vmm*>(v_k), x_delta, chunk_regs);

                        mov(reg_aux2, reg_state);
                        add(reg_aux2, chunk_offset);
                        store_vector_native_xf16(reg_aux2, const_cast<Vmm*>(v_h), chunk_regs);
                    }

                    // Output: out = dot(H, Q)
                    uni_vpxor(v_aux0, v_aux0, v_aux0);
                    for (int chunk = 0; chunk < num_chunks; chunk++) {
                        const int chunk_start = chunk * MAX_REGS_PER_VEC;
                        const int chunk_regs = std::min(MAX_REGS_PER_VEC, num_regs - chunk_start);
                        const size_t chunk_offset = chunk_start * XF16_ELEMS_PER_ZMM * m_jcp.data_prc.size();

                        mov(reg_aux2, reg_state);
                        add(reg_aux2, chunk_offset);
                        load_vector_native_xf16(const_cast<Vmm*>(v_h), reg_aux2, chunk_regs);

                        mov(reg_aux2, reg_query_tmp);
                        add(reg_aux2, chunk_offset);
                        load_vector_native_xf16(const_cast<Vmm*>(v_q), reg_aux2, chunk_regs);

                        accumulate_dot_product(const_cast<Vmm*>(v_h), const_cast<Vmm*>(v_q), chunk_regs);
                    }
                    uni_vpxor(x_out, x_out, x_out);
                    reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), x_out, x_tmp0, x_tmp1);
                }

                mov(reg_aux2, reg_out_seq);
                add(reg_aux2, static_cast<size_t>(v_idx * m_jcp.data_prc.size()));
                store(reg_aux2, Vmm(x_out.getIdx()), m_jcp.data_prc, 1);
            }
        }

        // Advance pointers using stride parameters.
        mov(reg_aux2, ptr[reg_args + GET_OFF(key_query_stride)]);
        lea(reg_key_seq, ptr[reg_key_seq + reg_aux2 * data_size]);
        lea(reg_query_seq, ptr[reg_query_seq + reg_aux2 * data_size]);

        mov(reg_aux2, ptr[reg_args + GET_OFF(value_stride)]);
        lea(reg_value_seq, ptr[reg_value_seq + reg_aux2 * data_size]);

        mov(reg_aux2, ptr[reg_args + GET_OFF(gate_beta_stride)]);
        lea(reg_gate_seq, ptr[reg_gate_seq + reg_aux2 * data_size]);
        lea(reg_beta_seq, ptr[reg_beta_seq + reg_aux2 * data_size]);

        mov(reg_aux2, ptr[reg_args + GET_OFF(output_stride)]);
        lea(reg_out_seq, ptr[reg_out_seq + reg_aux2 * data_size]);

        dec(reg_t);
        jnz(l_t_loop, T_NEAR);
    }

    L(l_end);

    this->postamble();

    exp_injector->prepare_table();
}

std::shared_ptr<JitKernelBase> create_gdn_jit_kernel(ov::element::Type data_prc,
                                                     size_t qk_head_size,
                                                     size_t v_tile,
                                                     bool fuse_qk_l2norm,
                                                     float q_l2_norm_eps,
                                                     float k_l2_norm_eps) {
    std::shared_ptr<JitKernelBase> res;
    jit_gdn_compile_params jcp;
    jcp.data_prc = data_prc;
    jcp.qk_head_size = qk_head_size;
    jcp.v_tile = v_tile;
    jcp.fuse_qk_l2norm = fuse_qk_l2norm;
    jcp.q_l2_norm_eps = q_l2_norm_eps;
    jcp.k_l2_norm_eps = k_l2_norm_eps;
    jcp.q_scale = 1.0F / std::sqrt(static_cast<float>(qk_head_size));

    if (data_prc != ov::element::bf16 && data_prc != ov::element::f16 && data_prc != ov::element::f32) {
        return res;
    }
    if (qk_head_size == 0) {
        return res;
    }
    if (v_tile == 0) {
        return res;
    }

    if ((data_prc == ov::element::bf16 || data_prc == ov::element::f16) && qk_head_size % 32 != 0) {
        return res;
    }

    if (data_prc == ov::element::bf16) {
        if (mayiuse(avx512_core_bf16)) {
            res = std::make_shared<jit_gdn_kernel<avx512_core_bf16>>(jcp);
        }
    } else if (data_prc == ov::element::f16) {
        if (mayiuse(avx512_core_fp16)) {
            res = std::make_shared<jit_gdn_kernel<avx512_core_fp16>>(jcp);
        }
    } else if (data_prc == ov::element::f32) {
        if (mayiuse(avx512_core)) {
            res = std::make_shared<jit_gdn_kernel<avx512_core>>(jcp);
        } else if (mayiuse(avx2)) {
            res = std::make_shared<jit_gdn_kernel<avx2>>(jcp);
        }
    }

    if (res) {
        res->create_kernel();
    }

    return res;
}

template struct jit_gdn_kernel<avx2>;
template struct jit_gdn_kernel<avx512_core>;
template struct jit_gdn_kernel<avx512_core_bf16>;
template struct jit_gdn_kernel<avx512_core_fp16>;

}  // namespace ov::intel_cpu::kernel
