// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"
#include "utils/bfloat16.hpp"

using namespace ov::intel_cpu::kernel;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak;
using Precision = typename InferenceEngine::Precision;

#define GET_OFF(field) offsetof(JitReduceCallArgs, field)
#define GET_OFF_POST(field) offsetof(JitReducePostCallArgs, field)


static inline bool isFloatCompatible(const Precision &type) {
    return Precision::FP32 == type || Precision::BF16 == type || Precision::FP64 == type;
}

///////////////////////////////
///// JitReduceKernelBase /////
///////////////////////////////

template<typename CallArgs>
void JitReduceKernelBase<CallArgs>::horiz_pd(const Xmm &xmm, const Operand &op) {
    switch (jcp.reduce_mode) {
        case Algorithm::ReduceAnd:
            uni_vandpd(xmm, xmm, op);
            break;
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
        case Algorithm::ReduceLogSumExp:
            uni_vaddpd(xmm, xmm, op);
            break;
        case Algorithm::ReduceMax:
            uni_vmaxpd(xmm, xmm, op);
            break;
        case Algorithm::ReduceMin:
            uni_vminpd(xmm, xmm, op);
            break;
        case Algorithm::ReduceOr:
            uni_vorpd(xmm, xmm, op);
            break;
        case Algorithm::ReduceProd:
            uni_vmulpd(xmm, xmm, op);
            break;
        default:
            IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
    }
}

////////// FLOAT 32 //////////
template<typename CallArgs>
void JitReduceKernelBase<CallArgs>::horiz_ps(const Xmm &vmm, const Operand &op) {
    switch (jcp.reduce_mode) {
        case Algorithm::ReduceAnd:
            uni_vandps(vmm, vmm, op);
            break;
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
        case Algorithm::ReduceLogSumExp:
            uni_vaddps(vmm, vmm, op);
            break;
        case Algorithm::ReduceMax:
            uni_vmaxps(vmm, vmm, op);
            break;
        case Algorithm::ReduceMin:
            uni_vminps(vmm, vmm, op);
            break;
        case Algorithm::ReduceOr:
            uni_vorps(vmm, vmm, op);
            break;
        case Algorithm::ReduceProd:
            uni_vmulps(vmm, vmm, op);
            break;
        default:
            IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
    }
}

template <typename CallArgs>
template <x64::cpu_isa_t isa>
void JitReduceKernelBase<CallArgs>::horiz_reduce_store_ps(const Xmm &vmm_dst, const Precision &dst_prc, bool load_embedded) {
    if (isa == x64::avx512_core) {
        auto zmm_dst = Zmm(vmm_dst.getIdx());
        auto ymm_dst = Ymm(vmm_dst.getIdx());

        vextractf64x4(ymm_aux1, zmm_dst, 1);
        horiz_ps(ymm_aux1, ymm_dst);
        vextractf128(xmm_aux2, ymm_aux1, 1);
        horiz_ps(xmm_aux1, xmm_aux2);
    } else if (isa == x64::avx2) {
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        vextractf128(xmm_aux1, ymm_aux1, 1);
        horiz_ps(xmm_aux1, xmm_dst);
    } else if (isa == x64::sse41) {
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        if (one_of(jcp.reduce_mode, Algorithm::ReduceL1, Algorithm::ReduceL2, Algorithm::ReduceLogSum, Algorithm::ReduceMean,
                                    Algorithm::ReduceSum, Algorithm::ReduceSumSquare, Algorithm::ReduceLogSumExp)) {
            uni_vhaddps(xmm_aux1, xmm_dst, xmm_dst);
            uni_vhaddps(xmm_aux1, xmm_aux1, xmm_aux1);
        } else {
            uni_vshufps(xmm_aux1, xmm_dst, xmm_dst, 0b00001110);
            horiz_ps(xmm_aux1, xmm_dst);
            uni_vshufps(xmm_aux2, xmm_aux1, xmm_aux1, 0b00000001);
            horiz_ps(xmm_aux1, xmm_aux2);
        }
    }

    if (isa != x64::sse41) {
        if (one_of(jcp.reduce_mode, Algorithm::ReduceL1, Algorithm::ReduceL2, Algorithm::ReduceLogSum, Algorithm::ReduceMean,
                                    Algorithm::ReduceSum, Algorithm::ReduceSumSquare, Algorithm::ReduceLogSumExp)) {
            uni_vhaddps(xmm_aux1, xmm_aux1, xmm_aux1);
            uni_vhaddps(xmm_aux1, xmm_aux1, xmm_aux1);
        } else {
            uni_vshufps(xmm_aux2, xmm_aux1, xmm_aux1, 0b00001110);
            horiz_ps(xmm_aux1, xmm_aux2);
            uni_vshufps(xmm_aux2, xmm_aux1, xmm_aux1, 0b00000001);
            horiz_ps(xmm_aux1, xmm_aux2);
        }
    }

    if (load_embedded) {
        if (isa == x64::avx512_core && exec_prc == dst_prc) {
            horiz_ps(xmm_aux1, ptr_b[reg_dst]);
        } else {
            loadScalar(xmm_aux2, ptr[reg_dst], exec_prc, dst_prc);
            horiz_ps(xmm_aux1, xmm_aux2);
        }
    }
    storeScalar(ptr[reg_dst], xmm_aux1, dst_prc, exec_prc);
}

////////// INTEGER 64 //////////
template<typename CallArgs>
void JitReduceKernelBase<CallArgs>::horiz_qq(const Xmm &vmm, const Operand &op) {
    switch (jcp.reduce_mode) {
        case Algorithm::ReduceAnd:
            uni_vandpd(vmm, vmm, op);
            break;
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
        case Algorithm::ReduceLogSumExp:
            uni_vpaddq(vmm, vmm, op);
            break;
        case Algorithm::ReduceMax:
            vpmaxsq(vmm, vmm, op);
            break;
        case Algorithm::ReduceMin:
            vpminsq(vmm, vmm, op);
            break;
        case Algorithm::ReduceOr:
            uni_vorpd(vmm, vmm, op);
            break;
        case Algorithm::ReduceProd:
            vpmullq(vmm, vmm, op);
            break;
        default:
            IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
    }
}

template <typename CallArgs>
template <x64::cpu_isa_t isa>
void JitReduceKernelBase<CallArgs>::horiz_reduce_store_qq(const Xmm &vmm_dst, const Precision &dst_prc, bool load_embedded) {
    if (isa == x64::avx512_core) {
        auto zmm_dst = Zmm(vmm_dst.getIdx());
        auto ymm_dst = Ymm(vmm_dst.getIdx());

        vextractf64x4(ymm_aux1, zmm_dst, 1);
        horiz_qq(ymm_aux1, ymm_dst);
        vextractf128(xmm_aux2, ymm_aux1, 1);
        horiz_qq(xmm_aux1, xmm_aux2);
        vshufpd(xmm_aux2, xmm_aux1, xmm_aux1, 0b00000001);
        horiz_qq(xmm_aux1, xmm_aux2);
    } else if (isa == x64::avx2) {
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        vextractf128(xmm_aux1, ymm_dst, 1);
        horiz_qq(xmm_aux1, xmm_dst);
        vshufpd(xmm_aux2, xmm_aux1, xmm_aux1, 0b00000001);
        horiz_qq(xmm_aux1, xmm_aux2);
    } else if (isa == x64::sse41) {
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        vshufpd(xmm_aux1, xmm_dst, xmm_dst, 0b00000001);
        horiz_qq(xmm_aux1, xmm_dst);
    }

    if (load_embedded) {
        if (isa == x64::avx512_core && exec_prc == dst_prc) {
            horiz_qq(xmm_aux1, ptr_b[reg_dst]);
        } else {
            loadScalar(xmm_aux2, ptr[reg_dst], exec_prc, dst_prc);
            horiz_qq(xmm_aux1, xmm_aux2);
        }
    }
    storeScalar(ptr[reg_dst], xmm_aux1, dst_prc, exec_prc);
}

///////////////////////////////
/////// JitReduceKernel ///////
///////////////////////////////
template <x64::cpu_isa_t isa>
JitReduceKernel<isa>::JitReduceKernel(const JitReduceConfigParams &jcp) : JitReduceKernelBase<JitReduceCallArgs>(jcp, jit_name()) {
    xmm_aux1 = Xmm(5);
    xmm_aux2 = Xmm(6);
    xmm_aux3 = Xmm(7);

    ymm_aux1 = Ymm(xmm_aux1.getIdx());
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::generate() {
    if (jcp.reduce_mode == Algorithm::ReduceLogSumExp) {
        exp_injector = std::make_shared<x64::jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f);
    }

    this->preamble();

    planar_layout = jcp.layout == ReduceLayoutType::reduce_ncsp || jcp.layout == ReduceLayoutType::reduce_nspc;

    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
    mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
    mov(reg_work_batch, ptr[reg_params + GET_OFF(work_batch)]);
    if (planar_layout) {
        mov(reg_reduce_w, ptr[reg_params + GET_OFF(reduce_w)]);
    }

    if (one_of(jcp.reduce_mode, Algorithm::ReduceAnd, Algorithm::ReduceL1, Algorithm::ReduceMax,
                                Algorithm::ReduceMin, Algorithm::ReduceProd, Algorithm::ReduceOr)) {
        mov(reg_table, l_table);
    }

    if (one_of(jcp.reduce_mode, Algorithm::ReduceAnd, Algorithm::ReduceOr)) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    }

    if (jcp.reduce_mode == Algorithm::ReduceOr) {
        uni_vmovups(vmm_aux, table_val(0));
    }

    reduce_main();
    reduce_tail();

    this->postamble();

    if (isa == x64::avx512_core) {
        vcvtneps2bf16->emit_data();
    }

    if (one_of(jcp.reduce_mode, Algorithm::ReduceAnd, Algorithm::ReduceL1, Algorithm::ReduceMax,
                                Algorithm::ReduceMin, Algorithm::ReduceProd, Algorithm::ReduceOr)) {
        prepare_aux_table();
    } else if (jcp.reduce_mode == Algorithm::ReduceLogSumExp) {
        exp_injector->prepare_table();
    }
}

template <x64::cpu_isa_t isa>
inline void JitReduceKernel<isa>::reduce_main() {
    // ================================================================
    // ***isa: AVX512***
    // ReduceAnd (Logical And)
    // step 1: init dst 0x3f800000 (1.0f)
    //              aux 0x3f800000 (1.0f)
    //             zero 0x00000000 (0.0f)
    // step 2: if src equals 0, set mask bit 0, else set mask bit 1
    // step 3: src = mask bit == 0 ? zero : aux
    // step 4: dst = dst & src
    //                  src    mask_bit    new_src    dst    new_dst
    //         case 1    ~0        1         1.0f     1.0f     1.0f
    //         case 2     0        0         0.0f     1.0f     0.0f
    //         case 3    ~0        1         1.0f     0.0f     0.0f
    //         case 4     0        0         0.0f     0.0f     0.0f
    // step 5: loop: offset src, and do step 2 and step 3
    //
    // ReduceOr (Logical Or)
    // step 1: init dst 0x00000000 (0.0f)
    //              aux 0x3f800000 (1.0f)
    //             zero 0x00000000 (0.0f)
    // step 2: if src equals 0, set mask bit 0, else set mask bit 1
    // step 3: src = mask bit == 0 ? zero : aux
    // step 4: dst = dst | src
    //                  src    mask_bit    new_src    dst    new_dst
    //         case 1     0        0         0.0f     0.0f     0.0f
    //         case 2    ~0        1         1.0f     0.0f     1.0f
    //         case 3     0        0         0.0f     1.0f     1.0f
    //         case 4    ~0        1         1.0f     1.0f     1.0f
    // step 5: loop: offset src, and do step 2 and step 3
    // ================================================================
    // ***isa: OTHER***
    // ReduceAnd (Logical And)
    // step 1: init dst 0x3f800000 (1.0f)
    // step 2: if src equals 0, set it 0x00000000, else set 0xffffffff
    // step 3: dst = dst & src
    //         0x3f800000 = 0x3f800000 & 0xffffffff (result: 1.0f)
    //         0x00000000 = 0x3f800000 & 0x00000000 (result: 0.0f)
    //         0x00000000 = 0x00000000 & 0xffffffff (result: 0.0f)
    //         0x00000000 = 0x00000000 & 0x00000000 (result: 0.0f)
    // step 4: loop: offset src, and do step 2 and step 3
    //
    // ReduceOr (Logical Or)
    // step 1: init dst 0x00000000 (0.0f)
    //              aux 0x3f800000 (1.0f)
    // step 2: dst = dst | src
    //         0x00000000 = 0x00000000 | 0x00000000
    //                  A = 0x00000000 | A
    //                  A =          A | 0x00000000
    //                  C =          A | B
    // (A, B stand for number other than 0x00000000)
    // step 3: loop: offset src, and do step 2
    // step 4: if dst equals 0, set it 0x00000000, else set 0xffffffff
    // step 5: dst = dst & aux
    //         0x00000000 = 0x00000000 & 0x3f800000 (result: 0.0f)
    //         0x3f800000 = 0xffffffff & 0x3f800000 (result: 1.0f)
    // ================================================================
    Label reduce_to_vector_label;
    Label reduce_to_scalar_label;
    Label reduce_to_gather_label;
    Label reduce_main_end_label;
    if (planar_layout) {
        cmp(reg_work_batch, 0);
        je(reduce_to_gather_label, T_NEAR);

        cmp(reg_reduce_w, 1); // planar layout reducing W
        je(reduce_to_scalar_label, T_NEAR);
    }

    // store vmm_dst directly into memory after reducing
    // cases: [planar layout reducing other dimensions but W] [blocked layout]
    L(reduce_to_vector_label);
    {
        int step = vlen / exec_prc.size() < 8 ? 8 : vlen / exec_prc.size();
        cmp(reg_work_amount, step);
        jl(reduce_main_end_label, T_NEAR); //avoid illegal loading and storing

        if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_prc == Precision::I64)) {
            uni_vmovups(vmm_aux, table_val(1));
        }

        // load
        load_dst_vector();

        // reduce
        reduce_kernel();

        // store
        store_dst_vector();

        jmp(reduce_main_end_label, T_NEAR);
    }

    // reduce vector in vmm_dst to be a scalar before store into memory
    // cases: [planar layout reducing W]
    L(reduce_to_scalar_label);
    {
        // init dst, dst loading is embedded in horiz_reduce_store
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
            case Algorithm::ReduceProd:
                uni_vmovups(vmm_dst, table_val(0));
                break;
            case Algorithm::ReduceL1:
                if (!(isa == x64::avx512_core && exec_prc == Precision::I64)) {
                    uni_vmovups(vmm_aux, table_val(1));
                }
                uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceLogSumExp:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceOr:
            case Algorithm::ReduceSum:
            case Algorithm::ReduceSumSquare:
                uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                break;
            case Algorithm::ReduceMax:
                if (isFloatCompatible(jcp.dst_prc)) {
                    uni_vmovups(vmm_dst, table_val(2));
                } else {
                    uni_vmovups(vmm_dst, table_val(4));
                }
                break;
            case Algorithm::ReduceMin:
                if (isFloatCompatible(jcp.dst_prc)) {
                    uni_vmovups(vmm_dst, table_val(3));
                } else {
                    uni_vmovups(vmm_dst, table_val(5));
                }
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
        // reduce
        reduce_main_loop();
        if (jcp.reduce_mode == Algorithm::ReduceOr && isa != x64::avx512_core) {
            uni_cmpneqps(vmm_dst, vmm_dst, vmm_zero);
            uni_vandps(vmm_dst, vmm_dst, vmm_aux);
        }
        // store
        // store after horizontal calculation and calculation with loaded original ptr[reg_dst]
        if (exec_prc == Precision::FP32) {
            horiz_reduce_store_ps<isa>(vmm_dst, jcp.dst_prc, true);
        } else if (exec_prc == Precision::FP64) {
            horiz_reduce_store_pd(vmm_dst, jcp.dst_prc, true);
        } else if (exec_prc == Precision::I64) {
            horiz_reduce_store_qq<isa>(vmm_dst, jcp.dst_prc, true);
        }

        jmp(reduce_main_end_label, T_NEAR);
    }

    // load vmm_src with gather, then store vmm_dst directly into memory after reducing
    // cases: [planar layout reducing small W]
    L(reduce_to_gather_label);
    {
        int step = 1;
        cmp(reg_work_amount, step);
        jl(reduce_main_end_label, T_NEAR); //avoid illegal loading and storing

        mov(reg_idx, ptr[reg_params + GET_OFF(idx)]);
        uni_vmovdqu(vmm_idx, ptr[reg_idx]);

        if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_prc == Precision::I64)) {
            uni_vmovups(vmm_aux, table_val(1));
        }

        // load
        load_dst_vector();

        // reduce
        Label reduce_loop_label;
        Label reduce_loop_end_label;
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            reduce_gather(vmm_dst, 0);
            if (isa == x64::sse41) {
                reduce_gather(vmm_dst_aux, 4 * jcp.src_prc.size());
            }

            add(reg_src, step * jcp.src_prc.size());
            sub(reg_work_amount, step);
            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);

        // store
        store_dst_vector();

        jmp(reduce_main_end_label, T_NEAR);
    }

    L(reduce_main_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_tail() {
    if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_prc == Precision::I64)) {
        uni_vmovups(xmm_aux, table_val(1));
    }

    Label tail_dst_shifted_label;
    Label tail_dst_fixed_label;
    Label reduce_tail_end_label;
    if (planar_layout) {
        cmp(reg_reduce_w, 1);  // planar layout reducing W
        je(tail_dst_fixed_label, T_NEAR);
    }

    // each src scalar reduce to each dst scalar (X1, X2, X3, ...) -> (Y1, Y2, Y3, ...)
    // cases: [planar layout reducing other dimensions but W] [blocked layout concern padding]
    L(tail_dst_shifted_label);
    {
        reduce_kernel_tail();

        jmp(reduce_tail_end_label, T_NEAR);
    }

    // each src scalar reduce to the same dst scalar (X1, X2, X3, ...) -> (Y1)
    // cases: [planar layout reducing W]
    L(tail_dst_fixed_label);
    {
        // load
        loadScalar(xmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);

        Label reduce_loop_label;
        Label reduce_loop_end_label;

        // reduce
        int step = 1;
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            loadScalar(xmm_src, ptr[reg_src], exec_prc, jcp.src_prc);

            reduce_kernel_scalar(xmm_src, xmm_dst);
            if (jcp.reduce_mode == Algorithm::ReduceOr) {
                if (exec_prc == Precision::FP32) {
                    uni_vcmpps(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
                    uni_vandps(xmm_dst, xmm_dst, xmm_aux);
                } else if (exec_prc == Precision::FP64 || exec_prc == Precision::I64) {
                    uni_vcmppd(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
                    uni_vandpd(xmm_dst, xmm_dst, xmm_aux);
                }
            }

            add(reg_src, step * jcp.src_prc.size());
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);

        // store
        storeScalar(ptr[reg_dst], xmm_dst, jcp.dst_prc, exec_prc);
    }

    L(reduce_tail_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::init_reg_reduce_stride() {
    mov(reg_reduce_stride, ptr[reg_params + GET_OFF(reduce_stride)]);
    mul_by_const(reg_reduce_stride, reg_tmp_64, jcp.src_prc.size());
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel() {
    Label reduce_label;
    Label reduce_end_label;
    Label reduce_batch_label;
    Label reduce_batch_end_label;

    const int step = vlen / exec_prc.size() < 8 ? 8 : vlen / exec_prc.size();
    cmp(reg_work_batch, 1);
    je(reduce_label, T_NEAR);

    init_reg_reduce_stride();

    L(reduce_batch_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_end_label, T_NEAR);

        reduce_batch();

        add(reg_src, step * jcp.src_prc.size());
        sub(reg_work_amount, step);
        jmp(reduce_batch_label, T_NEAR);
    }
    L(reduce_batch_end_label);

    L(reduce_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_end_label, T_NEAR);

        reduce_once();

        add(reg_src, step * jcp.src_prc.size());
        sub(reg_work_amount, step);
        jmp(reduce_label, T_NEAR);
    }
    L(reduce_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_once() {
    loadVector(vmm_src, ptr[reg_src], exec_prc, jcp.src_prc);
    reduce_kernel(vmm_src, vmm_dst);

    if (isa == x64::sse41) {
        loadVector(vmm_src, ptr[reg_src + 4 * jcp.src_prc.size()], exec_prc, jcp.src_prc);
        reduce_kernel(vmm_src, vmm_dst_aux);
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_batch() {
    mov(reg_src_aux, reg_src);
    mov(reg_work_batch_aux, reg_work_batch);

    Label reduce_batch_loop_label;
    Label reduce_batch_loop_end_label;
    L(reduce_batch_loop_label);
    {
        cmp(reg_work_batch_aux, 1);
        jl(reduce_batch_loop_end_label, T_NEAR);

        loadVector(vmm_src, ptr[reg_src_aux], exec_prc, jcp.src_prc);
        reduce_kernel(vmm_src, vmm_dst);
        if (isa == x64::sse41) {
            loadVector(vmm_src, ptr[reg_src_aux + 4 * jcp.src_prc.size()], exec_prc, jcp.src_prc);
            reduce_kernel(vmm_src, vmm_dst_aux);
        }

        add(reg_src_aux, reg_reduce_stride);
        sub(reg_work_batch_aux, 1);
        jmp(reduce_batch_loop_label, T_NEAR);
    }
    L(reduce_batch_loop_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_gather(const Vmm &vmm_dst, int64_t offset) {
    switch (jcp.src_prc) {
        case Precision::FP64:
        case Precision::I64:
            if (isa == x64::avx512_core) {
                kxnorq(k_mask, k_mask, k_mask);
                vgatherdpd(vmm_src | k_mask, ptr[reg_src + offset + ymm_idx]);
                if (jcp.src_prc == Precision::FP64 && exec_prc == Precision::I64) {
                    vcvtpd2qq(vmm_src, vmm_src);
                } else if (jcp.src_prc == Precision::I64 && exec_prc == Precision::FP64) {
                    vcvtqq2pd(vmm_src, vmm_src);
                }
            } else if (isa == x64::avx2) {
                uni_vpcmpeqq(vmm_mask, vmm_mask, vmm_mask);
                vgatherdpd(vmm_src, ptr[reg_src + offset + xmm_idx], vmm_mask);
            } else {
                pack_gathered_vector(vmm_src, vmm_idx, offset, jcp.src_prc);
            }
            break;
        case Precision::FP32:
        case Precision::I32:
            if (isa == x64::avx512_core) {
                kxnord(k_mask, k_mask, k_mask);
                if (jcp.src_prc == Precision::FP32) {
                    vgatherdps(vmm_src | k_mask, ptr[reg_src + offset + vmm_idx]);
                } else {
                    vpgatherdd(vmm_src | k_mask, ptr[reg_src + offset + vmm_idx]);
                    uni_vcvtdq2ps(vmm_src, vmm_src);
                }
            } else if (isa == x64::avx2) {
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                if (jcp.src_prc == Precision::FP32) {
                    vgatherdps(vmm_src, ptr[reg_src + offset + vmm_idx], vmm_mask);
                } else {
                    vpgatherdd(vmm_src, ptr[reg_src + offset + vmm_idx], vmm_mask);
                    uni_vcvtdq2ps(vmm_src, vmm_src);
                }
            } else {
                pack_gathered_vector(vmm_src, vmm_idx, offset, jcp.src_prc);
            }
            break;
        case Precision::BF16:
        case Precision::I8:
        case Precision::U8:
            pack_gathered_vector(vmm_src, vmm_idx, offset, jcp.src_prc);
            break;
        default:
            IE_THROW() << "Unkown source precision '" << jcp.src_prc << "'";
    }
    reduce_kernel(vmm_src, vmm_dst);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::pack_gathered_vector(const Vmm &vmm_val, const Vmm &vmm_index, int64_t offset, const Precision &src_prc) {
    sub(rsp, vlen);
    uni_vmovdqu(ptr[rsp], vmm_index);
    const size_t repeats = vlen / exec_prc.size();
    for (size_t i = 0; i < repeats; i++) {
        mov(reg_tmp_64.cvt32(), ptr[rsp + i * sizeof(int)]);
        Address table_idx = ptr[reg_src + offset + reg_tmp_64];

        switch (src_prc.size()) {
            case 8:
                mov(reg_tmp_64, table_idx);
                mov(ptr[rsp + i * sizeof(int64_t)], reg_tmp_64);
                break;
            case 4:
                mov(reg_tmp_64.cvt32(), table_idx);
                mov(ptr[rsp + i * sizeof(int32_t)], reg_tmp_64.cvt32());
                break;
            case 2:
                mov(reg_tmp_64.cvt16(), table_idx);
                mov(ptr[rsp + i * sizeof(ov::intel_cpu::bfloat16_t)], reg_tmp_64.cvt16());
                break;
            case 1:
                mov(reg_tmp_64.cvt8(), table_idx);
                mov(ptr[rsp + i * sizeof(char)], reg_tmp_64.cvt8());
                break;
            default:
                IE_THROW() << "Unkown source precision '" << src_prc << "'";
        }
    }

    switch (src_prc) {
        case Precision::FP64:
        case Precision::FP32:
        case Precision::I64:
        case Precision::I32:
            uni_vmovups(vmm_val, ptr[rsp]);
            break;
        case Precision::BF16:
            uni_vpmovzxwd(vmm_val, ptr[rsp]);
            uni_vpslld(vmm_val, vmm_val, 16);
        break;
        case Precision::I8:
            uni_vpmovsxbd(vmm_val, ptr[rsp]);
            break;
        case Precision::U8:
            uni_vpmovzxbd(vmm_val, ptr[rsp]);
            break;
        default:
            IE_THROW() << "Unkown source precision '" << src_prc << "'";
    }

    if (!isFloatCompatible(src_prc)) {
        uni_vcvtdq2ps(vmm_val, vmm_val);
    }
    add(rsp, vlen);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel_tail() {
    Label reduce_label;
    Label reduce_end_label;
    Label reduce_batch_label;
    Label reduce_batch_end_label;

    int step = 1;
    cmp(reg_work_batch, 1);
    je(reduce_label, T_NEAR);

    init_reg_reduce_stride();

    L(reduce_batch_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_end_label, T_NEAR);

        // load
        loadScalar(xmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);

        // reduce
        reduce_batch_tail();

        // store
        storeScalar(ptr[reg_dst], xmm_dst, jcp.dst_prc, exec_prc);

        add(reg_dst, step * jcp.dst_prc.size());
        add(reg_src, step * jcp.src_prc.size());
        sub(reg_work_amount, step);

        jmp(reduce_batch_label, T_NEAR);
    }
    L(reduce_batch_end_label);

    L(reduce_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_end_label, T_NEAR);

        // load
        loadScalar(xmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);

        // reduce
        reduce_batch_tail();

        // store
        storeScalar(ptr[reg_dst], xmm_dst, jcp.dst_prc, exec_prc);

        add(reg_dst, step * jcp.dst_prc.size());
        add(reg_src, step * jcp.src_prc.size());
        sub(reg_work_amount, step);

        jmp(reduce_label, T_NEAR);
    }
    L(reduce_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_once_tail() {
    loadScalar(xmm_src, ptr[reg_src], exec_prc, jcp.src_prc);
    reduce_kernel_scalar(xmm_src, xmm_dst);
    if (jcp.reduce_mode == Algorithm::ReduceOr) {
        if (exec_prc == Precision::FP32) {
            uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
            uni_vandps(xmm_dst, xmm_dst, xmm_aux);
        } else if (exec_prc == Precision::FP64 || exec_prc == Precision::I64) {
            uni_vcmppd(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
            uni_vandpd(xmm_dst, xmm_dst, xmm_aux);
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_batch_tail() {
    mov(reg_src_aux, reg_src);
    mov(reg_work_batch_aux, reg_work_batch);

    Label reduce_batch_loop_label;
    Label reduce_batch_loop_end_label;
    L(reduce_batch_loop_label);
    {
        cmp(reg_work_batch_aux, 1);
        jl(reduce_batch_loop_end_label, T_NEAR);

        loadScalar(xmm_src, ptr[reg_src_aux], exec_prc, jcp.src_prc);
        reduce_kernel_scalar(xmm_src, xmm_dst);
        if (jcp.reduce_mode == Algorithm::ReduceOr) {
            if (exec_prc == Precision::FP32) {
                uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
                uni_vandps(xmm_dst, xmm_dst, xmm_aux);
            } else if (exec_prc == Precision::FP64 || exec_prc == Precision::I64) {
                uni_vcmppd(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
                uni_vandpd(xmm_dst, xmm_dst, xmm_aux);
            }
        }

        add(reg_src_aux, reg_reduce_stride);
        sub(reg_work_batch_aux, 1);
        jmp(reduce_batch_loop_label, T_NEAR);
    }
    L(reduce_batch_loop_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_main_loop() {
    Label reduce_loop_label;
    Label reduce_loop_end_label;

    int step = vlen / exec_prc.size() < 8 ? 8 : vlen / exec_prc.size();
    L(reduce_loop_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_loop_end_label, T_NEAR);

        loadVector(vmm_src, ptr[reg_src], exec_prc, jcp.src_prc);
        reduce_kernel(vmm_src, vmm_dst);

        if (isa == x64::sse41) {
            loadVector(vmm_src, ptr[reg_src + 4 * jcp.src_prc.size()], exec_prc, jcp.src_prc);
            reduce_kernel(vmm_src, vmm_dst);
        }

        add(reg_src, step * jcp.src_prc.size());
        sub(reg_work_amount, step);

        jmp(reduce_loop_label, T_NEAR);
    }
    L(reduce_loop_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel(const Vmm &vmm_src, const Vmm &vmm_dst) {
    if (exec_prc == Precision::FP32) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == x64::avx512_core) {
                    vcmpps(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vmovups(vmm_dst | k_mask | T_z, vmm_dst);
                } else {
                    uni_cmpneqps(vmm_src, vmm_src, vmm_zero);
                    uni_vandps(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceL1:
                uni_vandps(vmm_src, vmm_src, vmm_aux);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMin:
                uni_vminps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                uni_vmulps(vmm_src, vmm_src, vmm_src);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == x64::avx512_core) {
                    vcmpps(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vorps(vmm_dst | k_mask, vmm_dst, vmm_aux);
                } else {
                    uni_vorps(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceProd:
                uni_vmulps(vmm_dst, vmm_dst, vmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    } else if (exec_prc == Precision::FP64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vandpd(vmm_dst | k_mask | T_z, vmm_dst, vmm_src);
                } else {
                    uni_vcmppd(vmm_src, vmm_src, vmm_zero, _cmp_neq_uq);
                    uni_vandpd(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceL1:
                uni_vandpd(vmm_src, vmm_src, vmm_aux);
                uni_vaddpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vaddpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMin:
                uni_vminpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                uni_vmulpd(vmm_src, vmm_src, vmm_src);
                uni_vaddpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);
                uni_vaddpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vblendmps(vmm_src | k_mask, vmm_zero, vmm_aux);
                }
                uni_vorpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceProd:
                uni_vmulpd(vmm_dst, vmm_dst, vmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    }  else if (exec_prc == Precision::I64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vmovups(vmm_dst | k_mask | T_z, vmm_dst);
                } else {
                    uni_vcmppd(vmm_src, vmm_src, vmm_zero, _cmp_neq_uq);
                    uni_vandpd(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceL1:
                if (isa == x64::avx512_core) {
                    vpabsq(vmm_src, vmm_src);
                } else {
                    uni_vandpd(vmm_src, vmm_src, vmm_aux);
                }
                uni_vpaddq(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vpaddq(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMax:
                if (isa == x64::avx512_core) {
                    vpmaxsq(vmm_dst, vmm_dst, vmm_src);
                } else {
                    // TODO
                }
                break;
            case Algorithm::ReduceMin:
                if (isa == x64::avx512_core) {
                    vpminsq(vmm_dst, vmm_dst, vmm_src);
                } else {
                    // TODO
                }
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                vpmullq(vmm_src, vmm_src, vmm_src);
                uni_vpaddq(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);
                uni_vpaddq(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vorpd(vmm_dst | k_mask, vmm_dst, vmm_aux);
                } else {
                    uni_vorpd(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceProd:
                vpmullq(vmm_dst, vmm_dst, vmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel_scalar(const Xmm &xmm_src, const Xmm &xmm_dst) {
    if (exec_prc == Precision::FP32) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                uni_cmpneqps(xmm_src, xmm_src, xmm_zero);
                uni_vandps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL1:
                uni_vandps(xmm_src, xmm_src, xmm_aux);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMin:
                uni_vminps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                uni_vmulps(xmm_src, xmm_src, xmm_src);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(xmm_src.getIdx(), xmm_src.getIdx() + 1);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceOr:
                uni_vorps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceProd:
                uni_vmulps(xmm_dst, xmm_dst, xmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    } else if (exec_prc == Precision::FP64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                uni_vcmppd(xmm_src, xmm_src, xmm_zero, _cmp_neq_uq);
                uni_vandpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL1:
                uni_vandpd(xmm_src, xmm_src, xmm_aux);
                uni_vaddpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vaddpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMin:
                uni_vminpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                uni_vmulpd(xmm_src, xmm_src, xmm_src);
                uni_vaddpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(xmm_src.getIdx(), xmm_src.getIdx() + 1);
                uni_vaddpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceOr:
                uni_vorpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceProd:
                uni_vmulpd(xmm_dst, xmm_dst, xmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    } else if (exec_prc == Precision::I64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                uni_vcmppd(xmm_src, xmm_src, xmm_zero, _cmp_neq_uq);
                uni_vandpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL1:
                if (isa == x64::avx512_core) {
                    vpabsq(xmm_src, xmm_src);
                } else {
                    uni_vandpd(xmm_src, xmm_src, xmm_aux);
                }
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMax:
                if (isa == x64::avx512_core) {
                    vpmaxsq(xmm_dst, xmm_dst, xmm_src);
                } else {
                    // TODO
                }
                break;
            case Algorithm::ReduceMin:
                if (isa == x64::avx512_core) {
                    vpminsq(xmm_dst, xmm_dst, xmm_src);
                } else {
                    // TODO
                }
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                vpmullq(xmm_src, xmm_src, xmm_src);
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(xmm_src.getIdx(), xmm_src.getIdx() + 1);
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceOr:
                uni_vorpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceProd:
                vpmullq(xmm_dst, xmm_dst, xmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::load_dst_vector() {
    loadVector(vmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);
    if (isa == x64::sse41) {
        loadVector(vmm_dst_aux, ptr[reg_dst + 4 * jcp.dst_prc.size()], exec_prc, jcp.dst_prc);
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::store_dst_vector() {
    if (jcp.reduce_mode == Algorithm::ReduceOr && isa != x64::avx512_core) {
        if (exec_prc == Precision::FP32) {
            uni_cmpneqps(vmm_dst, vmm_dst, vmm_zero);
            uni_vandps(vmm_dst, vmm_dst, vmm_aux);
        } else if (exec_prc == Precision::FP64 || exec_prc == Precision::I64) {
            uni_vcmppd(vmm_dst, vmm_dst, vmm_zero, _cmp_neq_uq);
            uni_vandpd(vmm_dst, vmm_dst, vmm_aux);
        }

        if (isa == x64::sse41) {
            uni_cmpneqps(vmm_dst_aux, vmm_dst_aux, vmm_zero);
            uni_vandps(vmm_dst_aux, vmm_dst_aux, vmm_aux);
        }
    }
    storeVector(ptr[reg_dst], vmm_dst, jcp.dst_prc, exec_prc);
    if (isa == x64::sse41) {
        storeVector(ptr[reg_dst + 4 * jcp.dst_prc.size()], vmm_dst_aux, jcp.dst_prc, exec_prc);
    }
}

////////// DOUBLE //////////
template <>
void JitReduceKernel<x64::avx512_core>::horiz_reduce_store_pd(const Zmm &zmm_dst, const Precision &dst_prc, bool load_embedded) {
    auto ymm_dst = Ymm(zmm_dst.getIdx());

    vextractf64x4(ymm_aux1, zmm_dst, 1);
    horiz_pd(ymm_aux1, ymm_dst);
    vextractf128(xmm_aux2, ymm_aux1, 1);
    horiz_pd(xmm_aux1, xmm_aux2);
    if (one_of(jcp.reduce_mode, Algorithm::ReduceL1, Algorithm::ReduceL2, Algorithm::ReduceLogSum, Algorithm::ReduceMean,
                                Algorithm::ReduceSum, Algorithm::ReduceSumSquare, Algorithm::ReduceLogSumExp)) {
        vhaddpd(xmm_aux1, xmm_aux1, xmm_aux1);
    } else {
        vshufpd(xmm_aux2, xmm_aux1, xmm_aux1, 0b00000001);
        horiz_pd(xmm_aux1, xmm_aux2);
    }
    if (load_embedded) {
        if (exec_prc == dst_prc) {
            horiz_pd(xmm_aux1, ptr_b[reg_dst]);
        } else {
            loadScalar(xmm_aux2, ptr[reg_dst], exec_prc, dst_prc);
            horiz_pd(xmm_aux1, xmm_aux2);
        }
    }
    storeScalar(ptr[reg_dst], xmm_aux1, dst_prc, exec_prc);
}

template <>
void JitReduceKernel<x64::avx2>::horiz_reduce_store_pd(const Vmm &vmm_dst, const Precision &dst_prc, bool load_embedded) {
    Ymm ymm_dst = Ymm(vmm_dst.getIdx());
    vextractf128(xmm_aux1, ymm_dst, 0);
    vextractf128(xmm_aux2, ymm_dst, 1);
    horiz_pd(xmm_aux1, xmm_aux2);
    horiz_store_pd(xmm_aux1, dst_prc, load_embedded);
}

template <>
void JitReduceKernel<x64::sse41>::horiz_reduce_store_pd(const Vmm &vmm_dst, const Precision &dst_prc, bool load_embedded) {
    horiz_store_pd(vmm_dst, dst_prc, load_embedded);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::horiz_store_pd(const Xmm &xmm_dst, const Precision &dst_prc, bool load_embedded) {
    uni_vmovshdup(xmm_aux3, xmm_dst);          // dst:1,2,3,4; aux3:2,2,4,4
    horiz_pd(xmm_dst, xmm_aux3);               // dst:f(1,2),f(2,2),f(3,4),f(4,4)
    uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_dst); // aux3:f(3,4),f(4,4),4,4
    horiz_pd(xmm_dst, xmm_aux3);               // dst:f(1,2,3,4),...
    if (load_embedded) {
        if (exec_prc == dst_prc) {
            horiz_pd(xmm_dst, ptr_b[reg_dst]);
        } else {
            loadScalar(xmm_aux3, ptr[reg_dst], exec_prc, dst_prc);
            horiz_pd(xmm_dst, xmm_aux3);
        }
    }
    storeScalar(ptr[reg_dst], xmm_dst, dst_prc, exec_prc);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::prepare_aux_table() {
    auto broadcast_int32 = [&](uint32_t val) {
        for (size_t d = 0; d < vlen / exec_prc.size(); ++d) {
            dd(val);
        }
    };
    auto broadcast_int64 = [&](uint64_t val) {
        for (size_t d = 0; d < vlen / exec_prc.size(); ++d) {
            dq(val);
        }
    };

    align(64);
    L(l_table);

    if (exec_prc == Precision::FP32) {
        broadcast_int32(aux_vals.float_one);
        broadcast_int32(aux_vals.float_abs);
        broadcast_int32(aux_vals.float_min);
        broadcast_int32(aux_vals.float_max);
        broadcast_int32(aux_vals.float_int32_min);
        broadcast_int32(aux_vals.float_int32_max);
    } else if (exec_prc == Precision::FP64) {
        broadcast_int64(aux_vals.double_one);
        broadcast_int64(aux_vals.double_abs);
        broadcast_int64(aux_vals.double_min);
        broadcast_int64(aux_vals.double_max);
        broadcast_int64(aux_vals.double_int64_min);
        broadcast_int64(aux_vals.double_int64_max);
    } else if (exec_prc == Precision::I64) {
        broadcast_int64(aux_vals.int64_one);
        broadcast_int64(aux_vals.int64_abs);
        broadcast_int64(aux_vals.int64_min);
        broadcast_int64(aux_vals.int64_max);
        broadcast_int64(aux_vals.int64_min);
        broadcast_int64(aux_vals.int64_max);
    }
}


///////////////////////////////
///// JitReducePostKernel /////
///////////////////////////////
template <x64::cpu_isa_t isa>
JitReducePostKernel<isa>::JitReducePostKernel(const JitReduceConfigParams &jcp, const dnnl_primitive_attr &attr)
        : JitReduceKernelBase<JitReducePostCallArgs>(jcp, jit_name()), attr(attr) {
    xmm_aux1 = Xmm(4);
    xmm_aux2 = Xmm(5);
    xmm_aux3 = Xmm(6);

    ymm_aux1 = Ymm(xmm_aux1.getIdx());
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::generate() {
    const auto &p = attr.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(std::make_shared<x64::jit_uni_eltwise_injector_f32<isa>>(
                    this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(std::make_shared<x64::jit_uni_depthwise_injector_f32<isa>>(
                    this, post_op));
        } else if (post_op.is_quantization()) {
            quantization_injectors.push_back(std::make_shared<x64::jit_uni_quantization_injector_f32<isa>>(
                    this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
        }
    }

    if (jcp.reduce_mode == Algorithm::ReduceLogSum || jcp.reduce_mode == Algorithm::ReduceLogSumExp) {
        log_injector = std::make_shared<x64::jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_log, 0.f, 0.f, 1.f);
    }

    this->preamble();

    planar_layout = jcp.layout == ReduceLayoutType::reduce_ncsp || jcp.layout == ReduceLayoutType::reduce_nspc;

    mov(reg_dst, ptr[reg_params + GET_OFF_POST(dst)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF_POST(work_amount)]);
    mov(reg_channel_size, ptr[reg_params + GET_OFF_POST(channel_size)]);
    mov(reg_divisor, ptr[reg_params + GET_OFF_POST(divisor)]);
    if (!planar_layout)
        mov(reg_reduce_c, ptr[reg_params + GET_OFF_POST(reduce_c)]);
    if (attr.post_ops_.len() != 0) {
        mov(reg_post_ops_data, ptr[reg_params + GET_OFF_POST(post_op_data)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF_POST(oc_off)]);
    }

    if (jcp.layout == ReduceLayoutType::reduce_blocked) {
        reduce_post_main();
    } else if (jcp.layout == ReduceLayoutType::reduce_nspc && attr.post_ops_.len() != 0) {
        // the tail of channel dimension should always be concerned during post ops fusing for nspc layout
        Label reduce_nspc_loop_label;
        Label reduce_nspc_loop_end_label;
        mov(reg_total_work_amount, reg_work_amount);
        L(reduce_nspc_loop_label);
        {
            cmp(reg_total_work_amount, 0);
            jle(reduce_nspc_loop_end_label, T_NEAR);

            mov(reg_oc_off, 0);
            mov(reg_work_amount, reg_channel_size);
            reduce_post_main();
            reduce_post_tail();

            sub(reg_total_work_amount, reg_channel_size);
            jmp(reduce_nspc_loop_label, T_NEAR);
        }
        L(reduce_nspc_loop_end_label);
    } else {
        reduce_post_main();
        reduce_post_tail();
    }

    this->postamble();

    if (isa == x64::avx512_core) {
        vcvtneps2bf16->emit_data();
    }

    if (one_of(jcp.reduce_mode, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        log_injector->prepare_table();
    }

    for (auto& inj : eltwise_injectors) {
        inj->prepare_table();
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::reduce_post_main() {
    Label reduce_channel_label;
    Label reduce_map_label;
    if (planar_layout) {
        jmp(reduce_map_label, T_NEAR);
    } else {
        cmp(reg_reduce_c, 1);
        jne(reduce_map_label, T_NEAR);
    }

    // further reduce channel block since reduce channel batch has already been reduced
    // (X1, X2, X3, X4, X5, X6, X7, X8) -> (Y1, N/A, N/A, N/A, N/A, N/A, N/A, N/A)
    // cases: [blocked layout reducing channel dimensions]
    L(reduce_channel_label);
    {
        Label reduce_loop_label;
        Label reduce_loop_end_label;

        int step = vlen / exec_prc.size() < 8 ? 8 : vlen / exec_prc.size();
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            // load
            loadVector(vmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);
            if (isa == x64::sse41) {
                loadVector(vmm_dst_aux, ptr[reg_dst + 4 * jcp.dst_prc.size()], exec_prc, jcp.dst_prc);
            }

            // reduce and store
            if (exec_prc == Precision::FP32) {
                horiz_reduce_store_ps<isa>(vmm_dst, jcp.dst_prc);
            } else if (exec_prc == Precision::FP64) {
                horiz_reduce_store_pd(vmm_dst, jcp.dst_prc);
            } else if (exec_prc == Precision::I64) {
                horiz_reduce_store_qq<isa>(vmm_dst, jcp.dst_prc);
            }
            if (isa == x64::sse41) {
                if (exec_prc == Precision::FP32) {
                    horiz_reduce_store_ps<isa>(vmm_dst_aux, jcp.dst_prc, true);
                } else if (exec_prc == Precision::FP64) {
                    horiz_reduce_store_pd(vmm_dst_aux, jcp.dst_prc, true);
                } else if (exec_prc == Precision::I64) {
                    horiz_reduce_store_qq<isa>(vmm_dst_aux, jcp.dst_prc, true);
                }
            }

            add(reg_dst, step * jcp.dst_prc.size());
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);

        mov(reg_dst, ptr[reg_params + GET_OFF_POST(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF_POST(work_amount)]);
    }

    // reduce map for value in dst memory
    // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean]
    L(reduce_map_label);
    {
        if (one_of(jcp.reduce_mode, Algorithm::ReduceL2, Algorithm::ReduceMean, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
            if (jcp.reduce_mode == Algorithm::ReduceMean) {
                if (exec_prc == Precision::FP32) {
                    uni_vbroadcastss(vmm_aux, ptr[reg_divisor]);
                } else if (exec_prc == Precision::FP64 || exec_prc == Precision::I64) {
                    auto zmm_aux = Zmm(vmm_aux.getIdx());
                    vbroadcastsd(zmm_aux, ptr[reg_divisor]);
                }
            }

            Label reduce_loop_label;
            Label reduce_loop_end_label;

            int step = vlen / exec_prc.size() < 8 ? 8 : vlen / exec_prc.size();
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                loadVector(vmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);
                reduce_map_kernel(vmm_dst);
                if (attr.post_ops_.len() != 0) {
                    apply_post_ops(jcp.dst_prc, jcp.layout == ReduceLayoutType::reduce_ncsp);
                }
                storeVector(ptr[reg_dst], vmm_dst, jcp.dst_prc, exec_prc);

                if (isa == x64::sse41) {
                    loadVector(vmm_dst, ptr[reg_dst + 4 * jcp.dst_prc.size()], exec_prc, jcp.dst_prc);
                    reduce_map_kernel(vmm_dst);
                    if (attr.post_ops_.len() != 0) {
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            add(reg_oc_off, 4 * exec_prc.size());
                        }
                        apply_post_ops(jcp.dst_prc, jcp.layout == ReduceLayoutType::reduce_ncsp);
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            sub(reg_oc_off, 4 * exec_prc.size());
                        }
                    }
                    storeVector(ptr[reg_dst + 4 * jcp.dst_prc.size()], vmm_dst, jcp.dst_prc, exec_prc);
                }

                add(reg_dst, step * jcp.dst_prc.size());
                if (jcp.layout == ReduceLayoutType::reduce_nspc && attr.post_ops_.len() != 0) {
                    add(reg_oc_off, step * exec_prc.size());
                }
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);
        } else {
            if (attr.post_ops_.len() != 0) {
                Label reduce_loop_label;
                Label reduce_loop_end_label;

                int step = vlen / exec_prc.size() < 8 ? 8 : vlen / exec_prc.size();
                L(reduce_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(reduce_loop_end_label, T_NEAR);

                    loadVector(vmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);
                    apply_post_ops(jcp.dst_prc, jcp.layout == ReduceLayoutType::reduce_ncsp);
                    storeVector(ptr[reg_dst], vmm_dst, jcp.dst_prc, exec_prc);

                    if (isa == x64::sse41) {
                        loadVector(vmm_dst, ptr[reg_dst + 4 * jcp.dst_prc.size()], exec_prc, jcp.dst_prc);
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            add(reg_oc_off, 4 * exec_prc.size());
                        }
                        apply_post_ops(jcp.dst_prc, jcp.layout == ReduceLayoutType::reduce_ncsp);
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            sub(reg_oc_off, 4 * exec_prc.size());
                        }
                        storeVector(ptr[reg_dst + 4 * jcp.dst_prc.size()], vmm_dst, jcp.dst_prc, exec_prc);
                    }

                    add(reg_dst, step * jcp.dst_prc.size());
                    if (jcp.layout == ReduceLayoutType::reduce_nspc && attr.post_ops_.len() != 0) {
                        add(reg_oc_off, step * exec_prc.size());
                    }
                    sub(reg_work_amount, step);

                    jmp(reduce_loop_label, T_NEAR);
                }
                L(reduce_loop_end_label);
            }
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::reduce_post_tail() {
    // reduce map for tail in dst memory
    // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean] in planar layout
    if (one_of(jcp.reduce_mode, Algorithm::ReduceL2, Algorithm::ReduceMean, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        if (jcp.reduce_mode == Algorithm::ReduceMean) {
            if (exec_prc == Precision::FP32) {
                uni_vbroadcastss(xmm_aux, ptr[reg_divisor]);
            } else if (exec_prc == Precision::FP64 || exec_prc == Precision::I64) {
                auto ymm_aux = Ymm(xmm_aux.getIdx());
                vbroadcastsd(ymm_aux, ptr[reg_divisor]);
            }
        }

        Label reduce_loop_label;
        Label reduce_loop_end_label;

        int step = 1;
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            // load
            loadScalar(xmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);

            // reduce
            reduce_map_kernel_scalar(xmm_dst);

            // store
            if (attr.post_ops_.len() != 0) {
                apply_post_ops(jcp.dst_prc, jcp.layout == ReduceLayoutType::reduce_ncsp);
            }
            storeScalar(ptr[reg_dst], xmm_dst, jcp.dst_prc, exec_prc);

            add(reg_dst, step * jcp.dst_prc.size());
            if (jcp.layout == ReduceLayoutType::reduce_nspc && attr.post_ops_.len() != 0) {
                add(reg_oc_off, step * exec_prc.size());
            }
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);
    } else {
        if (attr.post_ops_.len() != 0) {
            Label reduce_loop_label;
            Label reduce_loop_end_label;

            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                // load
                loadScalar(xmm_dst, ptr[reg_dst], exec_prc, jcp.dst_prc);

                // store
                apply_post_ops(jcp.dst_prc, jcp.layout == ReduceLayoutType::reduce_ncsp);
                storeScalar(ptr[reg_dst], xmm_dst, jcp.dst_prc, exec_prc);

                add(reg_dst, step * jcp.dst_prc.size());
                if (jcp.layout == ReduceLayoutType::reduce_nspc && attr.post_ops_.len() != 0) {
                    add(reg_oc_off, step * exec_prc.size());
                }
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::apply_post_ops(const Precision &dst_prc, bool is_broadcast) {
    const auto &p = attr.post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int quantization_inj_idx = 0;
    int post_ops_data_offset = 0;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
            add(reg_d_weights, reg_oc_off);

            depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                    vmm_dst.getIdx(), vmm_dst.getIdx() + 1, reg_d_weights, reg_d_weights, is_broadcast);

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == dnnl::impl::alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || isFloatCompatible(dst_prc) || i != p.len() - 1;

            int s_idx = vmm_dst.getIdx();

            quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
            quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

            quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
            quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

            if (do_dequantization) {
                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);
            }

            post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
            quantization_inj_idx++;
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::reduce_map_kernel(const Vmm &vmm_dst) {
    if (jcp.reduce_mode == Algorithm::ReduceMean) {
        if (exec_prc == Precision::FP32) {
            uni_vdivps(vmm_dst, vmm_dst, vmm_aux);
        } else if (exec_prc == Precision::FP64) {
            uni_vdivpd(vmm_dst, vmm_dst, vmm_aux);
        } else if (exec_prc == Precision::I64) {
            if (isa == x64::avx512_core) {
                vcvtqq2pd(vmm_dst, vmm_dst);
            }
            uni_vdivpd(vmm_dst, vmm_dst, vmm_aux);
            uni_vroundpd(vmm_dst, vmm_dst, 0x3); // Truncation
            if (isa == x64::avx512_core) {
                vcvtpd2qq(vmm_dst, vmm_dst);
            }
        }
    } else if (jcp.reduce_mode == Algorithm::ReduceL2) {
        if (exec_prc == Precision::FP32) {
            uni_vsqrtps(vmm_dst, vmm_dst);
        } else if (exec_prc == Precision::FP64) {
            uni_vsqrtpd(vmm_dst, vmm_dst);
        } else if (exec_prc == Precision::I64) {
            if (isa == x64::avx512_core) {
                vcvtqq2pd(vmm_dst, vmm_dst);
                uni_vsqrtpd(vmm_dst, vmm_dst);
                uni_vroundpd(vmm_dst, vmm_dst, 0x3); // Truncation
                vcvtpd2qq(vmm_dst, vmm_dst);
            }
        }
    } else if (one_of(jcp.reduce_mode, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        log_injector->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::reduce_map_kernel_scalar(const Xmm &xmm_dst) {
    if (jcp.reduce_mode == Algorithm::ReduceMean) {
        if (exec_prc == Precision::FP32) {
            uni_vdivps(xmm_dst, xmm_dst, xmm_aux);
        } else if (exec_prc == Precision::FP64) {
            uni_vdivpd(xmm_dst, xmm_dst, xmm_aux);
        } else if (exec_prc == Precision::I64) {
            if (isa == x64::avx512_core) {
                vcvtqq2pd(xmm_dst, xmm_dst);
                uni_vdivpd(xmm_dst, xmm_dst, xmm_aux);
                uni_vroundpd(xmm_dst, xmm_dst, 0x3); // Truncation
                vcvtpd2qq(xmm_dst, xmm_dst);
            }
        }
    } else if (jcp.reduce_mode == Algorithm::ReduceL2) {
        if (exec_prc == Precision::FP32) {
            uni_vsqrtps(xmm_dst, xmm_dst);
        } else if (exec_prc == Precision::FP64) {
            uni_vsqrtpd(xmm_dst, xmm_dst);
        } else if (exec_prc == Precision::I64) {
            if (isa == x64::avx512_core) {
                vcvtqq2pd(xmm_dst, xmm_dst);
                uni_vsqrtpd(xmm_dst, xmm_dst);
                uni_vroundpd(xmm_dst, xmm_dst, 0x3); // Truncation
                vcvtpd2qq(xmm_dst, xmm_dst);
            }
        }
    } else if (one_of(jcp.reduce_mode, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        log_injector->compute_vector_range(xmm_dst.getIdx(), xmm_dst.getIdx() + 1);
    }
}

template <>
void JitReducePostKernel<x64::avx512_core>::horiz_reduce_store_pd(const Zmm &zmm_dst, const Precision &dst_prc, bool load_embedded) {
    auto ymm_dst = Ymm(zmm_dst.getIdx());

    vextractf64x4(ymm_aux1, zmm_dst, 1);
    horiz_pd(ymm_aux1, ymm_dst);
    vextractf128(xmm_aux2, ymm_aux1, 1);
    horiz_pd(xmm_aux1, xmm_aux2);
    if (one_of(jcp.reduce_mode, Algorithm::ReduceL1, Algorithm::ReduceL2, Algorithm::ReduceLogSum, Algorithm::ReduceMean,
                                Algorithm::ReduceSum, Algorithm::ReduceSumSquare, Algorithm::ReduceLogSumExp)) {
        vhaddpd(xmm_aux1, xmm_aux1, xmm_aux1);
    } else {
        uni_vmovhlps(xmm_aux2, xmm_aux2, xmm_aux1);
        horiz_pd(xmm_aux1, xmm_aux2);
    }
    if (load_embedded) {
        if (exec_prc == dst_prc) {
            horiz_pd(xmm_aux1, ptr_b[reg_dst]);
        } else {
            loadScalar(xmm_aux2, ptr[reg_dst], exec_prc, dst_prc);
            horiz_pd(xmm_aux1, xmm_aux2);
        }
    }
    storeScalar(ptr[reg_dst], xmm_aux1, dst_prc, exec_prc);
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::horiz_reduce_store_pd(const Vmm &vmm_dst, const Precision &dst_prc, bool load_embedded) {
    Ymm ymm_dst = Ymm(vmm_dst.getIdx());
    vextractf128(xmm_aux1, ymm_dst, 0);
    vextractf128(xmm_aux2, ymm_dst, 1);
    horiz_ps(xmm_aux1, xmm_aux2);
}


template class JitReduceKernel<x64::avx512_core>;
template class JitReduceKernel<x64::avx2>;
template class JitReduceKernel<x64::sse41>;

template class JitReducePostKernel<x64::avx512_core>;
template class JitReducePostKernel<x64::avx2>;
template class JitReducePostKernel<x64::sse41>;
