// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"
#include "utils/bfloat16.hpp"
#include <ie_ngraph_utils.hpp>

using namespace ov::intel_cpu::kernel;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak;

#define GET_OFF(field) offsetof(JitReduceCallArgs, field)
#define GET_OFF_POST(field) offsetof(JitReducePostCallArgs, field)


static inline bool isFloatCompatible(const ov::element::Type& type) {
    return ov::intel_cpu::one_of(type, ov::element::f32, ov::element::bf16, ov::element::f64);
}

///////////////////////////////
///// JitReduceKernelBase /////
///////////////////////////////

template<typename CallArgs>
JitReduceKernelBase<CallArgs>::JitReduceKernelBase(const char* name, const JitReduceConfigParams& jcp, x64::cpu_isa_t isa)
        : JitKernel<JitReduceConfigParams, CallArgs>(name, jcp, isa) {
    exec_el_type = jcp.src_el_type;
    if (exec_el_type.size() <= 4) {
        exec_el_type = ov::element::f32;
    } else if (exec_el_type == ov::element::u64) {
        exec_el_type = ov::element::i64;
    }

    planar_layout = one_of(jcp.layout, ReduceLayoutType::reduce_ncsp, ReduceLayoutType::reduce_nspc);

    if (one_of(ov::element::bf16, exec_el_type, jcp.src_el_type, jcp.dst_el_type)) {
        this->vcvtneps2bf16 = std::make_shared<jit_uni_vcvtneps2bf16>(this, isa);
    }
    if (jcp.reduce_mode == Algorithm::ReduceMax) {
        max_emitter = std::make_shared<ov::intel_cpu::jit_maximum_emitter>(this, isa, InferenceEngine::details::convertPrecision(exec_el_type));
    }
    if (jcp.reduce_mode == Algorithm::ReduceMin) {
        min_emitter = std::make_shared<ov::intel_cpu::jit_minimum_emitter>(this, isa, InferenceEngine::details::convertPrecision(exec_el_type));
    }
    if (one_of(jcp.reduce_mode, Algorithm::ReduceL2, Algorithm::ReduceSumSquare, Algorithm::ReduceProd)) {
        mul_emitter = std::make_shared<ov::intel_cpu::jit_multiply_emitter>(this, isa, InferenceEngine::details::convertPrecision(exec_el_type));
    }
}

////////// FLOAT 32 //////////
template<typename CallArgs>
void JitReduceKernelBase<CallArgs>::horiz_ps(const Xmm& vmm_dst, const Operand& op) {
    switch (this->jcp.reduce_mode) {
        case Algorithm::ReduceAnd:
            this->uni_vandps(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
        case Algorithm::ReduceLogSumExp:
            this->uni_vaddps(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceMax:
            this->uni_vmaxps(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceMin:
            this->uni_vminps(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceOr:
            this->uni_vorps(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceProd:
            this->uni_vmulps(vmm_dst, vmm_dst, op);
            break;
        default:
            IE_THROW() << "Unsupported reduce mode '" << algToString(this->jcp.reduce_mode) << "'";
    }
}

template <typename CallArgs>
template <x64::cpu_isa_t isa>
void JitReduceKernelBase<CallArgs>::horiz_reduce_store_ps(const Xmm& vmm_dst, const ov::element::Type& dst_el_type, bool load_embedded) {
    auto xmm_aux_1 = RegistersPool::Reg<Xmm>(this->registersPool);
    auto xmm_aux_2 = RegistersPool::Reg<Xmm>(this->registersPool);

    if (isa == x64::avx512_core) {
        auto zmm_dst = Zmm(vmm_dst.getIdx());
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto ymm_aux_1 = Ymm(xmm_aux_1.getIdx());

        this->vextractf64x4(ymm_aux_1, zmm_dst, 1);
        this->horiz_ps(ymm_aux_1, ymm_dst);
        this->vextractf128(xmm_aux_2, ymm_aux_1, 1);
        this->horiz_ps(xmm_aux_1, xmm_aux_2);
    } else if (isa == x64::avx2) {
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        this->vextractf128(xmm_aux_1, ymm_dst, 1);
        this->horiz_ps(xmm_aux_1, xmm_dst);
    } else if (isa == x64::sse41) {
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        if (one_of(this->jcp.reduce_mode, Algorithm::ReduceL1, Algorithm::ReduceL2, Algorithm::ReduceLogSum, Algorithm::ReduceMean,
                                          Algorithm::ReduceSum, Algorithm::ReduceSumSquare, Algorithm::ReduceLogSumExp)) {
            this->uni_vhaddps(xmm_aux_1, xmm_dst, xmm_dst);
            this->uni_vhaddps(xmm_aux_1, xmm_aux_1, xmm_aux_1);
        } else {
            this->uni_vshufps(xmm_aux_1, xmm_dst, xmm_dst, 0b00001110);
            this->horiz_ps(xmm_aux_1, xmm_dst);
            this->uni_vshufps(xmm_aux_2, xmm_aux_1, xmm_aux_1, 0b00000001);
            this->horiz_ps(xmm_aux_1, xmm_aux_2);
        }
    }

    if (isa != x64::sse41) {
        if (one_of(this->jcp.reduce_mode, Algorithm::ReduceL1, Algorithm::ReduceL2, Algorithm::ReduceLogSum, Algorithm::ReduceMean,
                                          Algorithm::ReduceSum, Algorithm::ReduceSumSquare, Algorithm::ReduceLogSumExp)) {
            this->uni_vhaddps(xmm_aux_1, xmm_aux_1, xmm_aux_1);
            this->uni_vhaddps(xmm_aux_1, xmm_aux_1, xmm_aux_1);
        } else {
            this->uni_vshufps(xmm_aux_2, xmm_aux_1, xmm_aux_1, 0b00001110);
            this->horiz_ps(xmm_aux_1, xmm_aux_2);
            this->uni_vshufps(xmm_aux_2, xmm_aux_1, xmm_aux_1, 0b00000001);
            this->horiz_ps(xmm_aux_1, xmm_aux_2);
        }
    }

    auto trg_el_type = dst_el_type;
    Reg64 trg_ptr = reg_dst;
    if (this->jcp.fuse_low_precision && (post_reduce || post_ops_fusing)) {
        trg_el_type = ov::element::f32; // TODO i64 fusing
        trg_ptr = reg_src;
    }
    if (load_embedded) {
        if (isa == x64::avx512_core && exec_el_type == trg_el_type) {
            this->horiz_ps(xmm_aux_1, this->ptr_b[trg_ptr]);
        } else {
            this->load_scalar(xmm_aux_2, this->ptr[trg_ptr], exec_el_type, trg_el_type);
            this->horiz_ps(xmm_aux_1, xmm_aux_2);
        }
    }
    this->store_scalar(this->ptr[trg_ptr], xmm_aux_1, trg_el_type, exec_el_type);
}

////////// INTEGER 64 //////////
template<typename CallArgs>
template <x64::cpu_isa_t isa>
void JitReduceKernelBase<CallArgs>::horiz_qq(const Xmm& vmm_dst, const Operand& op) {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;

    switch (this->jcp.reduce_mode) {
        case Algorithm::ReduceAnd:
            this->uni_vandpd(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
        case Algorithm::ReduceLogSumExp:
            this->uni_vpaddq(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceMax:
            if (isa == x64::avx512_core) {
                this->vpmaxsq(vmm_dst, vmm_dst, op);
            } else {
                auto vmm_aux_0 = getVmm();
                if (op.isMEM()) {
                    max_emitter->emit_code({vmm_dst.getIdx()}, {vmm_dst.getIdx()}, {vmm_aux_0.getIdx()}, {op.getIdx()});
                } else {
                    max_emitter->emit_code({vmm_dst.getIdx(), op.getIdx()}, {vmm_dst.getIdx()}, {vmm_aux_0.getIdx()});
                }
            }
            break;
        case Algorithm::ReduceMin:
            if (isa == x64::avx512_core) {
                this->vpminsq(vmm_dst, vmm_dst, op);
            } else {
                auto vmm_aux_0 = getVmm();
                if (op.isMEM()) {
                    min_emitter->emit_code({vmm_dst.getIdx()}, {vmm_dst.getIdx()}, {vmm_aux_0.getIdx()}, {op.getIdx()});
                } else {
                    min_emitter->emit_code({vmm_dst.getIdx(), op.getIdx()}, {vmm_dst.getIdx()}, {vmm_aux_0.getIdx()});
                }
            }
            break;
        case Algorithm::ReduceOr:
            this->uni_vorpd(vmm_dst, vmm_dst, op);
            break;
        case Algorithm::ReduceProd:
            if (isa == x64::avx512_core) {
                this->vpmullq(vmm_dst, vmm_dst, op);
            } else {
                auto vmm_aux_0 = getVmm();
                auto vmm_aux_1 = getVmm();
                if (op.isMEM()) {
                    mul_emitter->emit_code({vmm_dst.getIdx()}, {vmm_dst.getIdx()}, {vmm_aux_0.getIdx(), vmm_aux_1.getIdx()}, {op.getIdx()});
                } else {
                    mul_emitter->emit_code({vmm_dst.getIdx(), op.getIdx()}, {vmm_dst.getIdx()}, {vmm_aux_0.getIdx(), vmm_aux_1.getIdx()});
                }
            }
            break;
        default:
            IE_THROW() << "Unsupported reduce mode '" << algToString(this->jcp.reduce_mode) << "'";
    }
}

template <typename CallArgs>
template <x64::cpu_isa_t isa>
void JitReduceKernelBase<CallArgs>::horiz_reduce_store_qq(const Xmm& vmm_dst, const ov::element::Type& dst_el_type, bool load_embedded) {
    auto xmm_aux_1 = RegistersPool::Reg<Xmm>(this->registersPool);
    auto xmm_aux_2 = RegistersPool::Reg<Xmm>(this->registersPool);

    if (isa == x64::avx512_core) {
        auto zmm_dst = Zmm(vmm_dst.getIdx());
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto ymm_aux_1 = Ymm(xmm_aux_1.getIdx());

        this->vextractf64x4(ymm_aux_1, zmm_dst, 1);
        this->horiz_qq<isa>(ymm_aux_1, ymm_dst);
        this->vextractf128(xmm_aux_2, ymm_aux_1, 1);
        this->horiz_qq<isa>(xmm_aux_1, xmm_aux_2);
        this->vshufpd(xmm_aux_2, xmm_aux_1, xmm_aux_1, 0b00000001);
        this->horiz_qq<isa>(xmm_aux_1, xmm_aux_2);
    } else if (isa == x64::avx2) {
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        this->vextractf128(xmm_aux_1, ymm_dst, 1);
        this->horiz_qq<isa>(xmm_aux_1, xmm_dst);
        this->vshufpd(xmm_aux_2, xmm_aux_1, xmm_aux_1, 0b00000001);
        this->horiz_qq<isa>(xmm_aux_1, xmm_aux_2);
    } else if (isa == x64::sse41) {
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        this->vshufpd(xmm_aux_1, xmm_dst, xmm_dst, 0b00000001);
        this->horiz_qq<isa>(xmm_aux_1, xmm_dst);
    }

    auto trg_el_type = dst_el_type;
    Reg64 trg_ptr = reg_dst;
    if (this->jcp.fuse_low_precision && (post_reduce || post_ops_fusing)) {
        trg_el_type = ov::element::f32; // TODO i64 fusing
        trg_ptr = reg_src;
    }
    if (load_embedded) {
        if (isa == x64::avx512_core && exec_el_type == trg_el_type) {
            this->horiz_qq<isa>(xmm_aux_1, this->ptr_b[trg_ptr]);
        } else {
            this->load_scalar(xmm_aux_2, this->ptr[trg_ptr], exec_el_type, trg_el_type);
            this->horiz_qq<isa>(xmm_aux_1, xmm_aux_2);
        }
    }
    this->store_scalar(this->ptr[trg_ptr], xmm_aux_1, trg_el_type, exec_el_type);
}

///////////////////////////////
/////// JitReduceKernel ///////
///////////////////////////////

template <x64::cpu_isa_t isa>
JitReduceKernel<isa>::JitReduceKernel(const JitReduceConfigParams &jcp) : JitReduceKernelBase<JitReduceCallArgs>(jit_name(), jcp, isa) {
    loop_step = vlen / exec_el_type.size();
    if (isa == x64::sse41) {
        loop_step *= 2;
    }

    if (jcp.reduce_mode == Algorithm::ReduceLogSumExp) {
        exp_injector = std::make_shared<x64::jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f);
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::generate() {
    this->preamble();

    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    reg_src         = getReg64();
    reg_dst         = getReg64();
    reg_work_amount = getReg64();
    reg_work_batch  = getReg64();
    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
    mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
    mov(reg_work_batch, ptr[reg_params + GET_OFF(work_batch)]);

    reg_reduce_stride = getReg64();
    v_src = getVmm();
    v_dst = getVmm();

    if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_el_type == ov::element::i64)) {
        v_abs_mask = getVmm();
    }
    if (isa == x64::sse41) {
        v_dst_aux = getVmm();
    }

    if (planar_layout) {
        reg_reduce_w = getReg64();
        mov(reg_reduce_w, ptr[reg_params + GET_OFF(reduce_w)]);
    }
    if (one_of(jcp.reduce_mode, Algorithm::ReduceAnd, Algorithm::ReduceL1, Algorithm::ReduceMax,
                                Algorithm::ReduceMin, Algorithm::ReduceProd, Algorithm::ReduceOr)) { // TODO ReduceProd ?
        reg_table = getReg64();
        mov(reg_table, l_table);
    }
    if (one_of(jcp.reduce_mode, Algorithm::ReduceAnd, Algorithm::ReduceOr)) {
        v_zero = getVmm();
        uni_vpxor(v_zero, v_zero, v_zero);
    }
    if (jcp.reduce_mode == Algorithm::ReduceOr) {
        v_ones = getVmm();
        uni_vmovups(v_ones, table_val(0));
    }

    reduce_main();
    reduce_tail();

    registersPool.reset();

    this->postamble();

    if (vcvtneps2bf16) {
        vcvtneps2bf16->emit_data();
    }
    if (max_emitter) {
        max_emitter->emit_data();
    }
    if (min_emitter) {
        min_emitter->emit_data();
    }
    if (mul_emitter) {
        mul_emitter->emit_data();
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
    Label reduce_to_scalar_label;
    Label reduce_to_gather_label;
    Label reduce_main_end_label;
    if (planar_layout) {
        cmp(reg_work_batch, 0);
        je(reduce_to_gather_label, T_NEAR);

        cmp(reg_reduce_w, 1); // planar layout reducing W
        je(reduce_to_scalar_label, T_NEAR);
    }

    // store v_dst directly into memory after reducing
    // cases: [planar layout reducing other dimensions but W] [blocked layout]
    {
        cmp(reg_work_amount, loop_step);
        jl(reduce_main_end_label, T_NEAR); // avoid illegal loading and storing

        if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_el_type == ov::element::i64)) {
            uni_vmovups(v_abs_mask, table_val(1));
        }

        load_dst_vector();

        reduce_kernel();

        if (jcp.reduce_mode == Algorithm::ReduceMean) {
            auto reg_can_divide = getReg64();
            auto reg_divider = getReg64();
            auto vmm_divider = getVmm();
            Label reduce_divide_end_label;

            mov(reg_can_divide, ptr[reg_params + GET_OFF(can_divide)]);
            cmp(reg_can_divide, 0);
            je(reduce_divide_end_label, T_NEAR);
            {
                mov(reg_divider, ptr[reg_params + GET_OFF(divisor)]);
                if (exec_el_type.size() == 4) {
                    uni_vbroadcastss(vmm_divider, ptr[reg_divider]);
                } else if (exec_el_type.size() == 8) {
                    uni_vbroadcastsd(vmm_divider, ptr[reg_divider]);
                }
                if (exec_el_type == ov::element::f32) {
                    uni_vdivps(v_dst, v_dst, vmm_divider);
                    if (isa == x64::sse41) {
                        uni_vdivps(v_dst_aux, v_dst_aux, vmm_divider);
                    }
                } else if (exec_el_type == ov::element::f64) {
                    uni_vdivpd(v_dst, v_dst, vmm_divider);
                    if (isa == x64::sse41) {
                        uni_vdivpd(v_dst_aux, v_dst_aux, vmm_divider);
                    }
                } else if (exec_el_type == ov::element::i64) {
                    if (isa == x64::avx512_core) {
                        vcvtqq2pd(v_dst, v_dst);
                    } else {
                        // TODO
                    }
                    uni_vdivpd(v_dst, v_dst, vmm_divider);
                    uni_vroundpd(v_dst, v_dst, 0x3); // Truncation
                    if (isa == x64::avx512_core) {
                        vcvtpd2qq(v_dst, v_dst);
                    } else {
                        // TODO
                    }
                    if (isa == x64::sse41) {
                        // cvt
                        uni_vdivpd(v_dst_aux, v_dst_aux, vmm_divider);
                        // cvt
                    }
                }
            }
            L(reduce_divide_end_label);
        }

        store_dst_vector();

        jmp(reduce_main_end_label, T_NEAR);
    }

    // reduce vector in v_dst to be a scalar before store into memory
    // cases: [planar layout reducing W]
    L(reduce_to_scalar_label);
    {
        // init dst, dst loading is embedded in horiz_reduce_store
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
            case Algorithm::ReduceProd:
                uni_vmovups(v_dst, table_val(0));
                break;
            case Algorithm::ReduceL1:
                if (!(isa == x64::avx512_core && exec_el_type == ov::element::i64)) {
                    uni_vmovups(v_abs_mask, table_val(1));
                }
                uni_vpxor(v_dst, v_dst, v_dst);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceLogSumExp:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceOr:
            case Algorithm::ReduceSum:
            case Algorithm::ReduceSumSquare:
                uni_vpxor(v_dst, v_dst, v_dst);
                break;
            case Algorithm::ReduceMax:
                if (isFloatCompatible(jcp.dst_el_type)) {
                    uni_vmovups(v_dst, table_val(2));
                } else {
                    uni_vmovups(v_dst, table_val(4));
                }
                break;
            case Algorithm::ReduceMin:
                if (isFloatCompatible(jcp.dst_el_type)) {
                    uni_vmovups(v_dst, table_val(3));
                } else {
                    uni_vmovups(v_dst, table_val(5));
                }
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
        // reduce
        reduce_main_loop();
        if (jcp.reduce_mode == Algorithm::ReduceOr && isa != x64::avx512_core) {
            uni_cmpneqps(v_dst, v_dst, v_zero);
            uni_vandps(v_dst, v_dst, v_ones);
        }
        // store
        // store after horizontal calculation and calculation with loaded original ptr[reg_dst]
        if (exec_el_type == ov::element::f32) {
            horiz_reduce_store_ps<isa>(v_dst, jcp.dst_el_type, true);
        } else if (exec_el_type == ov::element::i64) {
            horiz_reduce_store_qq<isa>(v_dst, jcp.dst_el_type, true);
        }

        jmp(reduce_main_end_label, T_NEAR);
    }

    // load v_src with gather, then store v_dst directly into memory after reducing
    // cases: [planar layout reducing small W]
    L(reduce_to_gather_label);
    {
        int step = 1;
        cmp(reg_work_amount, step);
        jl(reduce_main_end_label, T_NEAR); // Avoid illegal loading and storing.

        auto reg_idx = getReg64();
        v_idx = getVmm();
        mov(reg_idx, ptr[reg_params + GET_OFF(idx)]);
        uni_vmovdqu(v_idx, ptr[reg_idx]);

        if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_el_type == ov::element::i64)) {
            uni_vmovups(v_abs_mask, table_val(1));
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

            reduce_gather(v_dst, 0);
            if (isa == x64::sse41) {
                reduce_gather(v_dst_aux, 4 * jcp.src_el_type.size());
            }

            add(reg_src, step * jcp.src_el_type.size());
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
    if (jcp.reduce_mode == Algorithm::ReduceL1 && !(isa == x64::avx512_core && exec_el_type == ov::element::i64)) {
        auto xmm_abs_mask = Xmm(v_abs_mask.getIdx());
        uni_vmovups(xmm_abs_mask, table_val(1));
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
        auto xmm_dst = Xmm(v_dst.getIdx());
        auto xmm_src = Xmm(v_src.getIdx());

        // load
        load_scalar(xmm_dst, ptr[reg_dst], exec_el_type, jcp.dst_el_type);

        Label reduce_loop_label;
        Label reduce_loop_end_label;

        // reduce
        int step = 1;
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            load_scalar(xmm_src, ptr[reg_src], exec_el_type, jcp.src_el_type);

            reduce_kernel_scalar(xmm_src, xmm_dst);
            if (jcp.reduce_mode == Algorithm::ReduceOr) {
                auto xmm_ones = Xmm(v_ones.getIdx());
                auto xmm_zero = Xmm(v_zero.getIdx());

                if (exec_el_type == ov::element::f32) {
                    uni_vcmpps(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
                    uni_vandps(xmm_dst, xmm_dst, xmm_ones);
                } else if (exec_el_type == ov::element::f64 || exec_el_type == ov::element::i64) {
                    uni_vcmppd(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
                    uni_vandpd(xmm_dst, xmm_dst, xmm_ones);
                }
            }

            add(reg_src, step * jcp.src_el_type.size());
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);

        // store
        store_scalar(ptr[reg_dst], xmm_dst, jcp.dst_el_type, exec_el_type);
    }

    L(reduce_tail_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::init_reg_reduce_stride() {
    auto reg_tmp_64 = getReg64();
    mov(reg_reduce_stride, ptr[reg_params + GET_OFF(reduce_stride)]);
    mul_by_const(reg_reduce_stride, reg_tmp_64, jcp.src_el_type.size());
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel() {
    Label reduce_label;
    Label reduce_end_label;
    Label reduce_batch_label;

    cmp(reg_work_batch, 1);
    je(reduce_label, T_NEAR);

    init_reg_reduce_stride();

    L(reduce_batch_label);
    {
        cmp(reg_work_amount, loop_step);
        jl(reduce_end_label, T_NEAR);

        reduce_batch();

        add(reg_src, loop_step * jcp.src_el_type.size());
        sub(reg_work_amount, loop_step);
        jmp(reduce_batch_label, T_NEAR);
    }

    L(reduce_label);
    {
        cmp(reg_work_amount, loop_step);
        jl(reduce_end_label, T_NEAR);

        reduce_once();

        add(reg_src, loop_step * jcp.src_el_type.size());
        sub(reg_work_amount, loop_step);
        jmp(reduce_label, T_NEAR);
    }
    L(reduce_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_once() {
    load_vector(v_src, ptr[reg_src], exec_el_type, jcp.src_el_type);
    reduce_kernel(v_src, v_dst);

    if (isa == x64::sse41) {
        load_vector(v_src, ptr[reg_src + 4 * jcp.src_el_type.size()], exec_el_type, jcp.src_el_type);
        reduce_kernel(v_src, v_dst_aux);
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_batch() {
    auto reg_src_aux = getReg64();
    auto reg_work_batch_aux = getReg64();

    mov(reg_src_aux, reg_src);
    mov(reg_work_batch_aux, reg_work_batch);

    Label reduce_batch_loop_label;
    Label reduce_batch_loop_end_label;
    L(reduce_batch_loop_label);
    {
        cmp(reg_work_batch_aux, 1);
        jl(reduce_batch_loop_end_label, T_NEAR);

        load_vector(v_src, ptr[reg_src_aux], exec_el_type, jcp.src_el_type);
        reduce_kernel(v_src, v_dst);
        if (isa == x64::sse41) {
            load_vector(v_src, ptr[reg_src_aux + 4 * jcp.src_el_type.size()], exec_el_type, jcp.src_el_type);
            reduce_kernel(v_src, v_dst_aux);
        }

        add(reg_src_aux, reg_reduce_stride);
        sub(reg_work_batch_aux, 1);
        jmp(reduce_batch_loop_label, T_NEAR);
    }
    L(reduce_batch_loop_end_label);
}

template <>
void JitReduceKernel<x64::avx512_core>::reduce_gather(const Zmm& vmm_dst, int64_t offset) {
    switch (jcp.src_el_type.size()) {
        case 8: {
                auto ymm_idx = Ymm(v_idx.getIdx());

                kxnorq(k_mask, k_mask, k_mask);
                vgatherdpd((Zmm)v_src | k_mask, ptr[reg_src + offset + ymm_idx]);
                if (jcp.src_el_type == ov::element::f64 && exec_el_type == ov::element::i64) {
                    vcvtpd2qq(v_src, v_src);
                } else if (jcp.src_el_type == ov::element::i64 && exec_el_type == ov::element::f64) {
                    vcvtqq2pd(v_src, v_src);
                }
            }
            break;
        case 4: {
                kxnord(k_mask, k_mask, k_mask);
                vgatherdps((Zmm)v_src | k_mask, ptr[reg_src + offset + v_idx]);
                if (jcp.src_el_type == ov::element::i32) {
                    uni_vcvtdq2ps(v_src, v_src);
                }
            }
            break;
        case 2:
        case 1:
            pack_gathered_vector(v_src, v_idx, offset, jcp.src_el_type);
            break;
        default:
            IE_THROW() << "Unkown source element type '" << jcp.src_el_type << "'";
    }
    reduce_kernel(v_src, vmm_dst);
}

template <>
void JitReduceKernel<x64::avx2>::reduce_gather(const Ymm& vmm_dst, int64_t offset) {
    switch (jcp.src_el_type.size()) {
        case 8: {
                auto v_mask = getVmm();
                auto xmm_idx = Xmm(v_idx.getIdx());

                uni_vpcmpeqq(v_mask, v_mask, v_mask);
                vgatherdpd(v_src, ptr[reg_src + offset + xmm_idx], v_mask);
                if (exec_el_type == ov::element::i64) {
                    // TODO Convert pd tp qq (v_src, v_src);
                }
            }
            break;
        case 4: {
                auto v_mask = getVmm();

                uni_vpcmpeqd(v_mask, v_mask, v_mask);
                vgatherdps(v_src, ptr[reg_src + offset + v_idx], v_mask);
                if (jcp.src_el_type == ov::element::i32) {
                    uni_vcvtdq2ps(v_src, v_src);
                }
            }
            break;
        case 2:
        case 1:
            pack_gathered_vector(v_src, v_idx, offset, jcp.src_el_type);
            break;
        default:
            IE_THROW() << "Unkown source element type '" << jcp.src_el_type << "'";
    }
    reduce_kernel(v_src, vmm_dst);
}

template <>
void JitReduceKernel<x64::sse41>::reduce_gather(const Xmm& vmm_dst, int64_t offset) {
    pack_gathered_vector(v_src, v_idx, offset, jcp.src_el_type);
    reduce_kernel(v_src, vmm_dst);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::pack_gathered_vector(const Vmm& vmm_val, const Vmm& vmm_index, int64_t offset, const ov::element::Type& src_el_type) {
    sub(rsp, vlen);
    uni_vmovdqu(ptr[rsp], vmm_index);
    const size_t repeats = vlen / exec_el_type.size();
    auto reg_tmp_64 = getReg64();
    auto reg_tmp_32 = Reg32(reg_tmp_64.getIdx());
    auto reg_tmp_16 = Reg16(reg_tmp_64.getIdx());
    auto reg_tmp_8  = Reg8(reg_tmp_64.getIdx());
    for (size_t i = 0; i < repeats; i++) {
        mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);
        Address table_idx = ptr[reg_src + offset + reg_tmp_64];

        switch (src_el_type.size()) {
            case 8:
                mov(reg_tmp_64, table_idx);
                mov(ptr[rsp + i * sizeof(int64_t)], reg_tmp_64);
                break;
            case 4:
                mov(reg_tmp_32, table_idx);
                mov(ptr[rsp + i * sizeof(int32_t)], reg_tmp_32);
                break;
            case 2:
                mov(reg_tmp_16, table_idx);
                mov(ptr[rsp + i * sizeof(ov::intel_cpu::bfloat16_t)], reg_tmp_16);
                break;
            case 1:
                mov(reg_tmp_8, table_idx);
                mov(ptr[rsp + i * sizeof(char)], reg_tmp_8);
                break;
            default:
                IE_THROW() << "Unkown source element type '" << src_el_type << "'";
        }
    }

    switch (src_el_type) {
        case ov::element::f64:
        case ov::element::f32:
        case ov::element::i64:
        case ov::element::i32:
            uni_vmovups(vmm_val, ptr[rsp]);
            break;
        case ov::element::bf16:
            uni_vpmovzxwd(vmm_val, ptr[rsp]);
            uni_vpslld(vmm_val, vmm_val, 16);
        break;
        case ov::element::i8:
            uni_vpmovsxbd(vmm_val, ptr[rsp]);
            break;
        case ov::element::u8:
            uni_vpmovzxbd(vmm_val, ptr[rsp]);
            break;
        default:
            IE_THROW() << "Unkown source element type '" << src_el_type << "'";
    }

    if (!isFloatCompatible(src_el_type)) {
        uni_vcvtdq2ps(vmm_val, vmm_val); // TODO i64?
    }
    add(rsp, vlen);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel_tail() {
    Label reduce_label;
    Label reduce_end_label;
    Label reduce_batch_label;
    auto xmm_dst = Xmm(v_dst.getIdx());

    int step = 1;
    cmp(reg_work_batch, 1);
    je(reduce_label, T_NEAR);

    init_reg_reduce_stride();

    L(reduce_batch_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_end_label, T_NEAR);

        load_scalar(xmm_dst, ptr[reg_dst], exec_el_type, jcp.dst_el_type);

        reduce_batch_tail();

        store_scalar(ptr[reg_dst], xmm_dst, jcp.dst_el_type, exec_el_type);

        add(reg_dst, step * jcp.dst_el_type.size());
        add(reg_src, step * jcp.src_el_type.size());
        sub(reg_work_amount, step);

        jmp(reduce_batch_label, T_NEAR);
    }

    L(reduce_label);
    {
        cmp(reg_work_amount, step);
        jl(reduce_end_label, T_NEAR);

        load_scalar(xmm_dst, ptr[reg_dst], exec_el_type, jcp.dst_el_type);

        reduce_batch_tail();

        store_scalar(ptr[reg_dst], xmm_dst, jcp.dst_el_type, exec_el_type);

        add(reg_dst, step * jcp.dst_el_type.size());
        add(reg_src, step * jcp.src_el_type.size());
        sub(reg_work_amount, step);

        jmp(reduce_label, T_NEAR);
    }
    L(reduce_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_once_tail() {
    auto xmm_dst = Xmm(v_dst.getIdx());
    auto xmm_src = Xmm(v_src.getIdx());

    load_scalar(xmm_src, ptr[reg_src], exec_el_type, jcp.src_el_type);
    reduce_kernel_scalar(xmm_src, xmm_dst);
    if (jcp.reduce_mode == Algorithm::ReduceOr) {
        auto xmm_zero = Xmm(v_zero.getIdx());
        auto xmm_ones = Xmm(v_ones.getIdx());

        if (exec_el_type == ov::element::f32) {
            uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
            uni_vandps(xmm_dst, xmm_dst, xmm_ones);
        } else if (exec_el_type == ov::element::f64 || exec_el_type == ov::element::i64) {
            uni_vcmppd(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
            uni_vandpd(xmm_dst, xmm_dst, xmm_ones);
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_batch_tail() {
    auto reg_src_aux = getReg64();
    auto reg_work_batch_aux = getReg64();
    auto xmm_src = Xmm(v_src.getIdx());
    auto xmm_dst = Xmm(v_dst.getIdx());

    mov(reg_src_aux, reg_src);
    mov(reg_work_batch_aux, reg_work_batch);

    Label reduce_batch_loop_label;
    Label reduce_batch_loop_end_label;
    L(reduce_batch_loop_label);
    {
        cmp(reg_work_batch_aux, 1);
        jl(reduce_batch_loop_end_label, T_NEAR);

        load_scalar(xmm_src, ptr[reg_src_aux], exec_el_type, jcp.src_el_type);
        reduce_kernel_scalar(xmm_src, xmm_dst);
        if (jcp.reduce_mode == Algorithm::ReduceOr) {
            auto xmm_zero = Xmm(v_zero.getIdx());
            auto xmm_ones = Xmm(v_ones.getIdx());

            if (exec_el_type == ov::element::f32) {
                uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
                uni_vandps(xmm_dst, xmm_dst, xmm_ones);
            } else if (exec_el_type == ov::element::f64 || exec_el_type == ov::element::i64) {
                uni_vcmppd(xmm_dst, xmm_dst, xmm_zero, _cmp_neq_uq);
                uni_vandpd(xmm_dst, xmm_dst, xmm_ones);
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

    L(reduce_loop_label);
    {
        cmp(reg_work_amount, loop_step);
        jl(reduce_loop_end_label, T_NEAR);

        load_vector(v_src, ptr[reg_src], exec_el_type, jcp.src_el_type);
        reduce_kernel(v_src, v_dst);

        if (isa == x64::sse41) {
            load_vector(v_src, ptr[reg_src + 4 * jcp.src_el_type.size()], exec_el_type, jcp.src_el_type);
            reduce_kernel(v_src, v_dst);
        }

        add(reg_src, loop_step * jcp.src_el_type.size());
        sub(reg_work_amount, loop_step);

        jmp(reduce_loop_label, T_NEAR);
    }
    L(reduce_loop_end_label);
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel(const Vmm& vmm_src, const Vmm& vmm_dst) {
    const size_t src_idx = static_cast<size_t>(vmm_src.getIdx());
    const size_t dst_idx = static_cast<size_t>(vmm_dst.getIdx());

    if (exec_el_type == ov::element::f32) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == x64::avx512_core) {
                    vcmpps(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    vmovups(vmm_dst | k_mask | T_z, vmm_dst);
                } else {
                    uni_cmpneqps(vmm_src, vmm_src, v_zero);
                    uni_vandps(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceL1:
                uni_vandps(vmm_src, vmm_src, v_abs_mask);
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
                exp_injector->compute_vector_range(src_idx, src_idx + 1);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == x64::avx512_core) {
                    vcmpps(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    vorps(vmm_dst | k_mask, vmm_dst, v_ones);
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
    } else if (exec_el_type == ov::element::f64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    vandpd(vmm_dst | k_mask | T_z, vmm_dst, vmm_src);
                } else {
                    uni_vcmppd(vmm_src, vmm_src, v_zero, _cmp_neq_uq);
                    uni_vandpd(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceL1:
                uni_vandpd(vmm_src, vmm_src, v_abs_mask);
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
                exp_injector->compute_vector_range(src_idx, src_idx + 1);
                uni_vaddpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    vblendmps(vmm_src | k_mask, v_zero, v_ones);
                }
                uni_vorpd(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceProd:
                uni_vmulpd(vmm_dst, vmm_dst, vmm_src);
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    }  else if (exec_el_type == ov::element::i64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == x64::avx512_core) {
                    vcmppd(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    vmovups(vmm_dst | k_mask | T_z, vmm_dst);
                } else {
                    uni_vcmppd(vmm_src, vmm_src, v_zero, _cmp_neq_uq);
                    uni_vandpd(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceL1:
                if (isa == x64::avx512_core) {
                    vpabsq(vmm_src, vmm_src);
                } else {
                    uni_vandpd(vmm_src, vmm_src, v_abs_mask);
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
                    max_emitter->emit_code({dst_idx, src_idx}, {dst_idx});
                } else {
                    auto vmm_aux_0 = getVmm();
                    max_emitter->emit_code({dst_idx, src_idx}, {dst_idx}, {vmm_aux_0.getIdx()});
                }
                break;
            case Algorithm::ReduceMin:
                if (isa == x64::avx512_core) {
                    min_emitter->emit_code({dst_idx, src_idx}, {dst_idx});
                } else {
                    auto vmm_aux_0 = getVmm();
                    min_emitter->emit_code({dst_idx, src_idx}, {dst_idx}, {vmm_aux_0.getIdx()});
                }
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                if (isa == x64::avx512_core) {
                    mul_emitter->emit_code({src_idx, src_idx}, {src_idx});
                } else {
                    auto vmm_aux_0 = getVmm();
                    auto vmm_aux_1 = getVmm();
                    mul_emitter->emit_code({src_idx, src_idx}, {src_idx}, {vmm_aux_0.getIdx(), vmm_aux_1.getIdx()});
                }
                uni_vpaddq(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(src_idx, src_idx + 1);
                uni_vpaddq(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == x64::avx512_core) {
                    // vcmppd(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    // vblendmps(vmm_src | k_mask, v_zero, v_ones);
                    vcmppd(k_mask, vmm_src, v_zero, _cmp_neq_uq);
                    vorpd(vmm_dst | k_mask, vmm_dst, v_ones);
                } else {
                    uni_vorpd(vmm_dst, vmm_dst, vmm_src);
                }
                break;
            case Algorithm::ReduceProd:
                if (isa == x64::avx512_core) {
                    mul_emitter->emit_code({dst_idx, src_idx}, {dst_idx});
                } else {
                    auto vmm_aux_0 = getVmm();
                    auto vmm_aux_1 = getVmm();
                    mul_emitter->emit_code({dst_idx, src_idx}, {dst_idx}, {vmm_aux_0.getIdx(), vmm_aux_1.getIdx()});
                }
                break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::reduce_kernel_scalar(const Xmm& xmm_src, const Xmm& xmm_dst) {
    if (exec_el_type == ov::element::f32) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd: {
                    auto xmm_zero = Xmm(v_zero.getIdx());
                    uni_cmpneqps(xmm_src, xmm_src, xmm_zero);
                    uni_vandps(xmm_dst, xmm_dst, xmm_src);
                } break;
            case Algorithm::ReduceL1: {
                    auto xmm_abs_mask = Xmm(v_abs_mask.getIdx());
                    uni_vandps(xmm_src, xmm_src, xmm_abs_mask);
                    uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                } break;
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
    } else if (exec_el_type == ov::element::f64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd: {
                    auto xmm_zero = Xmm(v_zero.getIdx());
                    uni_vcmppd(xmm_src, xmm_src, xmm_zero, _cmp_neq_uq);
                    uni_vandpd(xmm_dst, xmm_dst, xmm_src);
                } break;
            case Algorithm::ReduceL1: {
                    auto xmm_abs_mask = Xmm(v_abs_mask.getIdx());
                    uni_vandpd(xmm_src, xmm_src, xmm_abs_mask);
                    uni_vaddpd(xmm_dst, xmm_dst, xmm_abs_mask);
                } break;
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
    } else if (exec_el_type == ov::element::i64) {
        switch (jcp.reduce_mode) {
            case Algorithm::ReduceAnd: {
                    auto xmm_zero = Xmm(v_zero.getIdx());
                    uni_vcmppd(xmm_src, xmm_src, xmm_zero, _cmp_neq_uq);
                    uni_vandpd(xmm_dst, xmm_dst, xmm_src);
                } break;
            case Algorithm::ReduceL1:
                if (isa == x64::avx512_core) {
                    vpabsq(xmm_src, xmm_src);
                } else {
                    auto xmm_abs_mask = Xmm(v_abs_mask.getIdx());
                    uni_vandpd(xmm_src, xmm_src, xmm_abs_mask);
                }
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMax: {
                    const size_t src_idx = static_cast<size_t>(xmm_src.getIdx()), dst_idx = static_cast<size_t>(xmm_dst.getIdx());
                    if (isa == x64::avx512_core) {
                        max_emitter->emit_code({dst_idx, src_idx}, {dst_idx});
                    } else {
                        auto vmm_aux_0 = getVmm();
                        max_emitter->emit_code({dst_idx, src_idx}, {dst_idx}, {vmm_aux_0.getIdx()});
                    }
                } break;
            case Algorithm::ReduceMin: {
                    const size_t src_idx = static_cast<size_t>(xmm_src.getIdx()), dst_idx = static_cast<size_t>(xmm_dst.getIdx());
                    if (isa == x64::avx512_core) {
                        min_emitter->emit_code({dst_idx, src_idx}, {dst_idx});
                    } else {
                        auto vmm_aux_0 = getVmm();
                        min_emitter->emit_code({dst_idx, src_idx}, {dst_idx}, {vmm_aux_0.getIdx()});
                    }
                } break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare: {
                    const size_t src_idx = static_cast<size_t>(xmm_src.getIdx());
                    if (isa == x64::avx512_core) {
                        mul_emitter->emit_code({src_idx, src_idx}, {src_idx});
                    } else {
                        auto vmm_aux_0 = getVmm();
                        auto vmm_aux_1 = getVmm();
                        mul_emitter->emit_code({src_idx, src_idx}, {src_idx}, {vmm_aux_0.getIdx(), vmm_aux_1.getIdx()});
                    }
                    uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                } break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(xmm_src.getIdx(), xmm_src.getIdx() + 1);
                uni_vpaddq(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceOr:
                uni_vorpd(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceProd: {
                    const size_t src_idx = static_cast<size_t>(xmm_src.getIdx()), dst_idx = static_cast<size_t>(xmm_dst.getIdx());
                    if (isa == x64::avx512_core) {
                        mul_emitter->emit_code({dst_idx, src_idx}, {dst_idx});
                    } else {
                        auto vmm_aux_0 = getVmm();
                        auto vmm_aux_1 = getVmm();
                        mul_emitter->emit_code({dst_idx, src_idx}, {dst_idx}, {vmm_aux_0.getIdx(), vmm_aux_1.getIdx()});
                    }
                } break;
            default:
                IE_THROW() << "Unsupported reduce mode '" << algToString(jcp.reduce_mode) << "'";
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::load_dst_vector() {
    load_vector(v_dst, ptr[reg_dst], exec_el_type, jcp.dst_el_type);
    if (isa == x64::sse41) {
        load_vector(v_dst_aux, ptr[reg_dst + 4 * jcp.dst_el_type.size()], exec_el_type, jcp.dst_el_type);
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::store_dst_vector() {
    if (jcp.reduce_mode == Algorithm::ReduceOr && isa != x64::avx512_core) {
        if (exec_el_type == ov::element::f32) {
            uni_cmpneqps(v_dst, v_dst, v_zero);
            uni_vandps(v_dst, v_dst, v_ones);
        } else if (exec_el_type == ov::element::f64 || exec_el_type == ov::element::i64) {
            uni_vcmppd(v_dst, v_dst, v_zero, _cmp_neq_uq);
            uni_vandpd(v_dst, v_dst, v_ones);
        }

        if (isa == x64::sse41) {
            uni_cmpneqps(v_dst_aux, v_dst_aux, v_zero);
            uni_vandps(v_dst_aux, v_dst_aux, v_ones);
        }
    }
    store_vector(ptr[reg_dst], v_dst, jcp.dst_el_type, exec_el_type);
    if (isa == x64::sse41) {
        store_vector(ptr[reg_dst + 4 * jcp.dst_el_type.size()], v_dst_aux, jcp.dst_el_type, exec_el_type);
    }
}

template <x64::cpu_isa_t isa>
void JitReduceKernel<isa>::prepare_aux_table() {
    auto broadcast_int32 = [&](uint32_t val) {
        for (size_t d = 0; d < vlen / exec_el_type.size(); ++d) {
            dd(val);
        }
    };
    auto broadcast_int64 = [&](uint64_t val) {
        for (size_t d = 0; d < vlen / exec_el_type.size(); ++d) {
            dq(val);
        }
    };

    align(64);
    L(l_table);

    if (exec_el_type == ov::element::f32) {
        broadcast_int32(aux_vals.float_one);
        broadcast_int32(aux_vals.float_abs);
        broadcast_int32(aux_vals.float_min);
        broadcast_int32(aux_vals.float_max);
        broadcast_int32(aux_vals.float_int32_min);
        broadcast_int32(aux_vals.float_int32_max);
    } else if (exec_el_type == ov::element::f64) {
        broadcast_int64(aux_vals.double_one);
        broadcast_int64(aux_vals.double_abs);
        broadcast_int64(aux_vals.double_min);
        broadcast_int64(aux_vals.double_max);
        broadcast_int64(aux_vals.double_int64_min);
        broadcast_int64(aux_vals.double_int64_max);
    } else if (exec_el_type == ov::element::i64) {
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
JitReducePostKernel<isa>::JitReducePostKernel(const JitReduceConfigParams& jcp, const dnnl_primitive_attr& attr)
        : JitReduceKernelBase<JitReducePostCallArgs>(jit_name(), jcp, isa), attr(attr) {
    post_reduce = one_of(jcp.reduce_mode, Algorithm::ReduceL2, Algorithm::ReduceMean, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp);
    post_ops_fusing = attr.post_ops_.len() != 0;

    loop_step = vlen / exec_el_type.size();
    if (isa == x64::sse41) {
        loop_step *= 2;
    }

    if (jcp.reduce_mode == Algorithm::ReduceLogSum || jcp.reduce_mode == Algorithm::ReduceLogSumExp) {
        log_injector = std::make_shared<x64::jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_log, 0.f, 0.f, 1.f);
    }

    if (jcp.reduce_mode == Algorithm::ReduceMean) {
        division_emitter = std::make_shared<ov::intel_cpu::jit_divide_emitter>(this, isa, InferenceEngine::details::convertPrecision(exec_el_type));
        division_emitter->second_is_float = true;
    }
    if (jcp.reduce_mode == Algorithm::ReduceL2) {
        sqrt_emitter = std::make_shared<ov::intel_cpu::jit_sqrt_emitter>(this, isa, InferenceEngine::details::convertPrecision(exec_el_type));
        sqrt_emitter->rounding_type = jit_emitter::RoundType::truncation;
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::generate() {
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    const auto &p = attr.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(std::make_shared<x64::jit_uni_eltwise_injector_f32<isa>>(
                    this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
        } else if (post_op.is_depthwise()) {
           if (!reg_d_weights.isInitialized()) {
               reg_d_weights = getReg64();
           }
            depthwise_injectors.push_back(std::make_shared<x64::jit_uni_depthwise_injector_f32<isa>>(
                    this, post_op));
        } else if (post_op.is_quantization()) {
           if (!reg_d_weights.isInitialized()) {
               reg_d_weights = getReg64();
           }
           if (!reg_d_bias.isInitialized()) {
               reg_d_bias = getReg64();
           }
           if (!v_d_weights.isInitialized()) {
               v_d_weights = getVmm();
           }
           if (!v_d_bias.isInitialized()) {
               v_d_bias = getVmm();
           }
            quantization_injectors.push_back(std::make_shared<x64::jit_uni_quantization_injector_f32<isa>>(
                    this, post_op, v_d_weights, v_d_bias, reg_d_weights, reg_d_bias));
        }
    }

    this->preamble();

    reg_dst         = getReg64();
    reg_work_amount = getReg64();
    mov(reg_dst, ptr[reg_params + GET_OFF_POST(dst)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF_POST(work_amount)]);

    v_dst = getVmm();

    if (!planar_layout) {
        reg_reduce_c = getReg64();
        mov(reg_reduce_c, ptr[reg_params + GET_OFF_POST(reduce_c)]);
    }
    if (post_ops_fusing) {
        reg_oc_off        = getReg64();
        reg_post_ops_data = getReg64();
        mov(reg_post_ops_data, ptr[reg_params + GET_OFF_POST(post_op_data)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF_POST(oc_off)]);
    }
    if (jcp.reduce_mode == Algorithm::ReduceMean) {
        v_divisor = getVmm();
        reg_divisor = getReg64();
        mov(reg_divisor, ptr[reg_params + GET_OFF_POST(divisor)]);
    }
    if (jcp.fuse_low_precision) {
        reg_src = getReg64();
        mov(reg_src, ptr[reg_params + GET_OFF_POST(src)]);
    }

    if (jcp.layout == ReduceLayoutType::reduce_blocked) {
        reduce_post_main();
    } else if (jcp.layout == ReduceLayoutType::reduce_nspc && post_ops_fusing) {
        auto reg_channel_size      = getReg64();
        auto reg_total_work_amount = getReg64();
        // the tail of channel dimension should always be concerned during post ops fusing for nspc layout
        Label reduce_nspc_loop_label;
        Label reduce_nspc_loop_end_label;
        mov(reg_channel_size, ptr[reg_params + GET_OFF_POST(channel_size)]);
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

    registersPool.reset();

    this->postamble();

    if (vcvtneps2bf16) {
        vcvtneps2bf16->emit_data();
    }
    if (max_emitter) {
        max_emitter->emit_data();
    }
    if (min_emitter) {
        min_emitter->emit_data();
    }
    if (mul_emitter) {
        mul_emitter->emit_data();
    }
    if (division_emitter) {
        division_emitter->emit_data();
    }
    if (sqrt_emitter) {
        sqrt_emitter->emit_data();
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
    {
        Label reduce_loop_label;
        Label reduce_loop_end_label;
        RegistersPool::Reg<Vmm> v_dst_aux;
        if (isa == x64::sse41) {
            v_dst_aux = getVmm();
        }

        // int step = vlen / exec_el_type.size() < 8 ? 8 : vlen / exec_el_type.size();
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, loop_step);
            jl(reduce_loop_end_label, T_NEAR);

            // load
            wrap_load_vector(v_dst, exec_el_type, jcp.dst_el_type, 0);
            if (isa == x64::sse41) {
                wrap_load_vector(v_dst_aux, exec_el_type, jcp.dst_el_type, 4);
            }

            // reduce and store
            if (exec_el_type == ov::element::f32) {
                horiz_reduce_store_ps<isa>(v_dst, jcp.dst_el_type);
            } else if (exec_el_type == ov::element::i64) {
                horiz_reduce_store_qq<isa>(v_dst, jcp.dst_el_type);
            }
            if (isa == x64::sse41) {
                if (exec_el_type == ov::element::f32) {
                    horiz_reduce_store_ps<isa>(v_dst_aux, jcp.dst_el_type, true);
                } else if (exec_el_type == ov::element::i64) {
                    horiz_reduce_store_qq<isa>(v_dst_aux, jcp.dst_el_type, true);
                }
            }

            add(reg_dst, loop_step * jcp.dst_el_type.size());
            if (jcp.fuse_low_precision) {
                add(reg_src, loop_step * sizeof(float)); // TODO i64 fusing
            }
            sub(reg_work_amount, loop_step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);

        if (post_reduce || post_ops_fusing) {
            mov(reg_dst, ptr[reg_params + GET_OFF_POST(dst)]);
            if (jcp.fuse_low_precision)
                mov(reg_src, ptr[reg_params + GET_OFF_POST(src)]);
            mov(reg_work_amount, ptr[reg_params + GET_OFF_POST(work_amount)]);
        }
    }

    // reduce map for value in dst memory
    // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean]
    L(reduce_map_label);
    {
        if (post_reduce) {
            if (jcp.reduce_mode == Algorithm::ReduceMean) {
                if (exec_el_type.size() == 4) {
                    uni_vbroadcastss(v_divisor, ptr[reg_divisor]);
                } else if (exec_el_type.size() == 8) {
                    uni_vbroadcastsd(v_divisor, ptr[reg_divisor]);
                }
            }

            Label reduce_loop_label;
            Label reduce_loop_end_label;

            L(reduce_loop_label);
            {
                cmp(reg_work_amount, loop_step);
                jl(reduce_loop_end_label, T_NEAR);

                wrap_load_vector(v_dst, exec_el_type, jcp.dst_el_type, 0);
                reduce_map_kernel(v_dst);
                if (post_ops_fusing) {
                    apply_post_ops(jcp.dst_el_type, jcp.layout == ReduceLayoutType::reduce_ncsp);
                }
                store_vector(ptr[reg_dst], v_dst, jcp.dst_el_type, exec_el_type);

                if (isa == x64::sse41) {
                    wrap_load_vector(v_dst, exec_el_type, jcp.dst_el_type, 4);
                    reduce_map_kernel(v_dst);
                    if (post_ops_fusing) {
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            add(reg_oc_off, 4 * exec_el_type.size());
                        }
                        apply_post_ops(jcp.dst_el_type, jcp.layout == ReduceLayoutType::reduce_ncsp);
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            sub(reg_oc_off, 4 * exec_el_type.size());
                        }
                    }
                    store_vector(ptr[reg_dst + 4 * jcp.dst_el_type.size()], v_dst, jcp.dst_el_type, exec_el_type);
                }

                add(reg_dst, loop_step * jcp.dst_el_type.size());
                if (jcp.layout == ReduceLayoutType::reduce_nspc && post_ops_fusing) {
                    add(reg_oc_off, loop_step * exec_el_type.size());
                }
                sub(reg_work_amount, loop_step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);
        } else {
            if (post_ops_fusing) {
                Label reduce_loop_label;
                Label reduce_loop_end_label;

                L(reduce_loop_label);
                {
                    cmp(reg_work_amount, loop_step);
                    jl(reduce_loop_end_label, T_NEAR);

                    load_vector(v_dst, ptr[reg_dst], exec_el_type, jcp.dst_el_type);
                    apply_post_ops(jcp.dst_el_type, jcp.layout == ReduceLayoutType::reduce_ncsp);
                    store_vector(ptr[reg_dst], v_dst, jcp.dst_el_type, exec_el_type);

                    if (isa == x64::sse41) {
                        load_vector(v_dst, ptr[reg_dst + 4 * jcp.dst_el_type.size()], exec_el_type, jcp.dst_el_type);
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            add(reg_oc_off, 4 * exec_el_type.size());
                        }
                        apply_post_ops(jcp.dst_el_type, jcp.layout == ReduceLayoutType::reduce_ncsp);
                        if (jcp.layout != ReduceLayoutType::reduce_ncsp) {
                            sub(reg_oc_off, 4 * exec_el_type.size());
                        }
                        store_vector(ptr[reg_dst + 4 * jcp.dst_el_type.size()], v_dst, jcp.dst_el_type, exec_el_type);
                    }

                    add(reg_dst, loop_step * jcp.dst_el_type.size());
                    if (jcp.fuse_low_precision) {
                        add(reg_src, loop_step * sizeof(float)); //TODO i64
                    }
                    if (jcp.layout == ReduceLayoutType::reduce_nspc && post_ops_fusing) {
                        add(reg_oc_off, loop_step * exec_el_type.size());
                    }
                    sub(reg_work_amount, loop_step);

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
    auto xmm_dst = Xmm(v_dst.getIdx());
    if (one_of(jcp.reduce_mode, Algorithm::ReduceL2, Algorithm::ReduceMean, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        if (jcp.reduce_mode == Algorithm::ReduceMean) {
            auto xmm_divisor = Xmm(v_divisor.getIdx());
            if (exec_el_type.size() == 4) {
                uni_vbroadcastss(xmm_divisor, ptr[reg_divisor]);
            } else if (exec_el_type.size() == 8) {
                auto ymm_aux = Ymm(xmm_divisor.getIdx());
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

            wrap_load_scalar(xmm_dst, exec_el_type, jcp.dst_el_type, 0);

            reduce_map_kernel_scalar(xmm_dst);

            if (post_ops_fusing) {
                apply_post_ops(jcp.dst_el_type, jcp.layout == ReduceLayoutType::reduce_ncsp);
            }
            store_scalar(ptr[reg_dst], xmm_dst, jcp.dst_el_type, exec_el_type);

            add(reg_dst, step * jcp.dst_el_type.size());
            if (jcp.fuse_low_precision) {
                add(reg_src, step * sizeof(float)); // TODO i64
            }
            if (jcp.layout == ReduceLayoutType::reduce_nspc && post_ops_fusing) {
                add(reg_oc_off, step * exec_el_type.size());
            }
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);
    } else {
        if (post_ops_fusing) {
            Label reduce_loop_label;
            Label reduce_loop_end_label;

            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                wrap_load_scalar(xmm_dst, exec_el_type, jcp.dst_el_type, 0);

                apply_post_ops(jcp.dst_el_type, jcp.layout == ReduceLayoutType::reduce_ncsp);
                store_scalar(ptr[reg_dst], xmm_dst, jcp.dst_el_type, exec_el_type);

                add(reg_dst, step * jcp.dst_el_type.size());
                if (jcp.fuse_low_precision) {
                    add(reg_src, step * sizeof(float)); // TODO i64
                }
                if (jcp.layout == ReduceLayoutType::reduce_nspc && post_ops_fusing) {
                    add(reg_oc_off, step * exec_el_type.size());
                }
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);
        }
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::apply_post_ops(const ov::element::Type& dst_el_type, bool is_broadcast) {
    const auto &p = attr.post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int quantization_inj_idx = 0;
    int post_ops_data_offset = 0;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(v_dst.getIdx(), v_dst.getIdx() + 1);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
            add(reg_d_weights, reg_oc_off);

            depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                    v_dst.getIdx(), v_dst.getIdx() + 1, reg_d_weights, reg_d_weights, is_broadcast);

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == dnnl::impl::alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || isFloatCompatible(dst_el_type) || i != p.len() - 1;

            int s_idx = v_dst.getIdx();

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
void JitReducePostKernel<isa>::reduce_map_kernel(const Vmm& vmm_dst) {
    if (jcp.reduce_mode == Algorithm::ReduceMean) {
        division_emitter->emit_code({ vmm_dst.getIdx(), v_divisor.getIdx() }, { vmm_dst.getIdx() });
    } else if (jcp.reduce_mode == Algorithm::ReduceL2) {
        sqrt_emitter->emit_code({ vmm_dst.getIdx() }, { vmm_dst.getIdx() });
    } else if (one_of(jcp.reduce_mode, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        log_injector->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::reduce_map_kernel_scalar(const Xmm& xmm_dst) {
    if (jcp.reduce_mode == Algorithm::ReduceMean) {
        division_emitter->emit_code({ xmm_dst.getIdx(), v_divisor.getIdx() }, { xmm_dst.getIdx() });
    } else if (jcp.reduce_mode == Algorithm::ReduceL2) {
        sqrt_emitter->emit_code({ xmm_dst.getIdx() }, { xmm_dst.getIdx() });
    } else if (one_of(jcp.reduce_mode, Algorithm::ReduceLogSum, Algorithm::ReduceLogSumExp)) {
        log_injector->compute_vector_range(xmm_dst.getIdx(), xmm_dst.getIdx() + 1);
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::wrap_load_vector(const Vmm& vmm_val, const element::Type& dst_dt, const element::Type& src_dt, size_t offset) {
    if (jcp.fuse_low_precision) {
        load_vector(vmm_val, ptr[reg_src + offset * sizeof(float)], dst_dt, src_dt); // TODO i64 fusing
    } else {
        load_vector(vmm_val, ptr[reg_dst + offset * dst_dt.size()], dst_dt, src_dt);
    }
}

template <x64::cpu_isa_t isa>
void JitReducePostKernel<isa>::wrap_load_scalar(const Xmm& xmm_val, const element::Type& dst_dt, const element::Type& src_dt, size_t offset) {
    if (jcp.fuse_low_precision) {
        load_scalar(xmm_val, ptr[reg_src + offset * sizeof(float)], dst_dt, src_dt); // TODO i64 fusing
    } else {
        load_scalar(xmm_val, ptr[reg_dst + offset * dst_dt.size()], dst_dt, src_dt);
    }
}


template class JitReduceKernel<x64::avx512_core>;
template class JitReduceKernel<x64::avx2>;
template class JitReduceKernel<x64::sse41>;

template class JitReducePostKernel<x64::avx512_core>;
template class JitReducePostKernel<x64::avx2>;
template class JitReducePostKernel<x64::sse41>;
