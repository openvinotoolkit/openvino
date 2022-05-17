/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "jit_generator.hpp"
#include "jit_uni_fork_softmax_kernel_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_softmax_call_s, field)

template<cpu_isa_t isa>
jit_uni_fork_softmax_kernel_f32<isa>::jit_uni_fork_softmax_kernel_f32(jit_softmax_conf_t ajpp) : jpp(ajpp) {
    if (jpp.dt == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
        bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_zmm_1, bf16_emu_zmm_2, bf16_emu_zmm_3,
                                             bf16_emu_gpr, bf16_emu_zmm_4, bf16_emu_zmm_5));
    }
}


template <cpu_isa_t isa>
status_t jit_uni_fork_softmax_kernel_f32<isa>::init_conf(jit_softmax_conf_t &jpp,
                   const softmax_desc_t &pd, const memory_desc_wrapper &src_d,
                   const memory_desc_wrapper &dst_d) {
    auto ndims = pd.data_desc.ndims;
    auto dims = pd.data_desc.dims;
    auto axis = pd.softmax_axis;

    jpp.dt = src_d.data_type();
    jpp.dt_size = src_d.data_type_size();

    size_t nregs = cpu_isa_traits<isa>::n_vregs;

    if (jpp.dt == data_type::bf16) {
        if (isa != avx512_common) {
            return status::unimplemented;
        }
        else if (!mayiuse(avx512_core_bf16)) {
            nregs -= 5; // reserved for the bf16 emulator
        }
    }

    size_t aux_simd_registers = 5; // 3 aux for exp + one + (-FTL_MAX)
    size_t regs_for_one_unroll = 2;
    size_t max_inner_unroll = (nregs - aux_simd_registers) / regs_for_one_unroll;
    size_t max_channels_unroll = 4;

    jpp.outer_size = utils::array_product(dims, axis);
    jpp.channels = dims[axis];
    jpp.inner_size = utils::array_product(dims + axis + 1, ndims - axis - 1);

    if (jpp.outer_size < 1 || jpp.channels < 1 || jpp.inner_size < 1) {
        return status::unimplemented;
    }

    jpp.ur_inner = max_inner_unroll;
    jpp.ur_channel = nstl::min(max_channels_unroll, jpp.channels);
    jpp.outer_block = 2 * cpu_isa_traits<isa>::vlen / sizeof(float);

    if (jpp.inner_size == 1) {
        // limit max jit code size for dense case
        if (jpp.channels > 128) {
            return status::unimplemented;
        }

        // ref implementation is faster for small work amount
        if (jpp.channels * jpp.outer_size < 16) {
            return status::unimplemented;
        }
    }

    return status::success;
}

template <cpu_isa_t isa>
int jit_uni_fork_softmax_kernel_f32<isa>::id_vreg_max(int ur_inner) {
    return 5+ur_inner;
}

template <cpu_isa_t isa>
int jit_uni_fork_softmax_kernel_f32<isa>::id_vreg_denom(int ur_inner) {
    return 5+jpp.ur_inner + ur_inner;
}

template <cpu_isa_t isa>
int jit_uni_fork_softmax_kernel_f32<isa>::id_vreg_src(int ur_inner) {
    return 5+2*jpp.ur_inner;
}

template <cpu_isa_t isa>
auto jit_uni_fork_softmax_kernel_f32<isa>::vreg_max(int ur_inner) -> Vmm {
    return Vmm(id_vreg_max(ur_inner));
}

template <cpu_isa_t isa>
auto jit_uni_fork_softmax_kernel_f32<isa>::vreg_denom(int ur_inner) -> Vmm {
    return Vmm(id_vreg_denom(ur_inner));
}

template <cpu_isa_t isa>
auto jit_uni_fork_softmax_kernel_f32<isa>::vreg_src(int ur_inner) -> Vmm {
    return Vmm(id_vreg_src(ur_inner));
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::load_vector(Vmm vmm_src_, const Xbyak::Address &op) {
    switch (jpp.dt) {
        case data_type::f32:
            uni_vmovups(vmm_src_, op);
            break;
        case data_type::bf16:
            uni_vpmovzxwd(vmm_src_, op);
            uni_vpslld(vmm_src_, vmm_src_, 16);
            break;
        default:
            assert(!"unknown data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::load_scalar(Xmm xmm_src_, const Xbyak::Address &op) {
    switch (jpp.dt) {
        case data_type::f32:
            movss(xmm_src_, op);
            break;
        case data_type::bf16:
            pinsrw(xmm_src_, op, 0x0);
            uni_vpslld(xmm_src_, xmm_src_, 16);
            break;
        default:
            assert(!"unknown data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::store_vector(const Xbyak::Address &op, Vmm vmm_dst_) {
    Ymm ymm_dst_ = Ymm(vmm_dst_.getIdx());
    switch (jpp.dt) {
        case data_type::f32:
            uni_vmovups(op, vmm_dst_);
            break;
        case data_type::bf16:
            if (mayiuse(avx512_core_bf16))
                vcvtneps2bf16(ymm_dst_, vmm_dst_);
            else
                bf16_emu_->vcvtneps2bf16(ymm_dst_, Zmm(vmm_dst_.getIdx()));
            vmovdqu16(op, ymm_dst_);
            break;
        default:
            assert(!"unknown data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::store_scalar(const Xbyak::Address &op, Xmm xmm_dst_) {
    switch (jpp.dt) {
        case data_type::f32:
            movss(op, xmm_dst_);
            break;
        case data_type::bf16:
            if (mayiuse(avx512_core_bf16))
                vcvtneps2bf16(xmm_dst_, xmm_dst_);
            else
                bf16_emu_->vcvtneps2bf16(xmm_dst_, Ymm(xmm_dst_.getIdx()));
            pextrw(op, xmm_dst_, 0x0);
            break;
        default:
            assert(!"unknown dst_dt");
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynom
            0x3f800001, // [5] p0 = 1.0000001f
            0x3efffe85, // [6] p2 = 0.4999887f
            0x3e2aaa3e, // [7] p3 = 0.16666505f
            0x3d2bb1b1, // [8] p4 = 0.041917507f
            0x3c091ec1, // [9] p5 = 0.008369149f
            0x42b0c0a5, //[10] max logf = 88.3762589f
            0xc1766666  //[11] min logf = -14.5f
    };

    align(64);
    L(l_table);
    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) {
            dd(cvals[i]);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::simd_expf(const Vmm &vmm_src) {
    uni_vminps(vmm_src, vmm_src, ptr[imm_addr64 + 10 * vlen]);
    uni_vmaxps(vmm_src, vmm_src, ptr[imm_addr64 + 11 * vlen]);
    uni_vmovups(vmm_aux0, vmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    uni_vmulps(vmm_src, vmm_src, ptr[imm_addr64 + 2 * vlen]);
    uni_vaddps(vmm_src, vmm_src, ptr[imm_addr64 + 1 * vlen]);

    // tmp = floorf(fx)
    if (isa < avx512_common) {
        uni_vroundps(vmm_aux1, vmm_src, _op_floor);
    } else {
        vcvtps2dq(vmm_aux1 | T_rd_sae, vmm_src);
        vcvtdq2ps(vmm_aux1, vmm_aux1);

        vcmpps(k_mask_tmp, vmm_aux1, vmm_src, _cmp_gt_os);
        vmovups(vmm_aux2 | k_mask_tmp | T_z, zword[imm_addr64 + 0 * vlen]);

        uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux2);
    }
    //keep fx for further computations
    uni_vmovups(vmm_src, vmm_aux1); //vmm_src = fx
    // compute 2^n
    uni_vcvtps2dq(vmm_aux2, vmm_src);
    uni_vpaddd(vmm_aux2, vmm_aux2, ptr[imm_addr64 + 4 * vlen]);
    uni_vpslld(vmm_aux2, vmm_aux2, 23); //Vmm(6) = 2^-fx

    //x = x - fx * ln2
    uni_vfnmadd231ps(vmm_aux0, vmm_aux1, ptr[imm_addr64 + 3 * vlen]);
    // y = p5
    uni_vmovups(vmm_src, ptr[imm_addr64 + 9 * vlen]);
    // y = y * x + p4
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[imm_addr64 + 8 * vlen]);
    // y = y * x + p3
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[imm_addr64 + 7 * vlen]);
    // y = y * x + p2
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[imm_addr64 + 6 * vlen]);
    // y = y * x + p1
    uni_vfmadd213ps(vmm_src, vmm_aux0, vmm_one);
    // y = y * x + p0
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[imm_addr64 + 5 * vlen]);  //exp(q)
    // y = y * 2^n
    uni_vmulps(vmm_src, vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::scalar_expf(const Xmm &xmm_src) {
    minss(xmm_src, ptr[imm_addr64 + 10 * vlen]);
    maxss(xmm_src, ptr[imm_addr64 + 11 * vlen]);
    movups(xmm_aux0, xmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    mulss(xmm_src, ptr[imm_addr64 + 2 * vlen]);
    addss(xmm_src, ptr[imm_addr64 + 1 * vlen]);
    // tmp = floorf(fx)
    roundss(xmm_aux1, xmm_src, _op_floor);
    //keep fx for further computations
    movups(xmm_src, xmm_aux1); //xmm_src = fx
    // compute 2^n
    cvtps2dq(xmm_aux2, xmm_src);
    paddd(xmm_aux2, ptr[imm_addr64 + 4 * vlen]);
    pslld(xmm_aux2, 23); //Xmm(6) = 2^-fx

    //calculation fx * ln2
    mulss(xmm_aux1, ptr[imm_addr64 + 3 * vlen]);
    //x = x - fx * ln2
    subss(xmm_aux0, xmm_aux1);
    // y = p5
    movups(xmm_src, ptr[imm_addr64 + 9 * vlen]);
    // y = y * x + p4
    mulss(xmm_src, xmm_aux0);
    addss(xmm_src, ptr[imm_addr64 + 8 * vlen]);

    // y = y * x + p3
    mulss(xmm_src, xmm_aux0);
    addss(xmm_src, ptr[imm_addr64 + 7 * vlen]);
    // y = y * x + p2
    mulss(xmm_src, xmm_aux0);
    addss(xmm_src, ptr[imm_addr64 + 6 * vlen]);

    // y = y * x + p1
    mulss(xmm_src, xmm_aux0);
    addss(xmm_src, xmm_one);

    // y = y * x + p0
    mulss(xmm_src, xmm_aux0);
    addss(xmm_src, ptr[imm_addr64 + 5 * vlen]); //exp(q)

    // y = y * 2^n
    mulps(xmm_src, xmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::simd_loop_max(int ur_inner) {
    Label loop_channel_blocks;
    Label loop_channel_tail;
    Label loop_channel_end;

    for (int i = 0; i < ur_inner; ++i) {
        uni_vbroadcastss(vreg_max(i), xmm_float_min);
    }

    mov(reg_ch_work, reg_channels);
    mov(reg_src_ptr, reg_src_base_ptr);

    L(loop_channel_blocks); {
        cmp(reg_ch_work, jpp.ur_channel);
        jl(loop_channel_tail, T_NEAR);

        for (int i = 0; i < ur_inner; ++i) {
            for (int c = 0; c < (int)jpp.ur_channel; ++c) {
                load_vector(vreg_src(i), ptr[reg_src_ptr + (i*simd_w + c*jpp.inner_size) * jpp.dt_size]);
                uni_vmaxps(vreg_max(i), vreg_max(i), vreg_src(i));
            }
        }

        sub(reg_ch_work, jpp.ur_channel);
        add(reg_src_ptr, jpp.ur_channel * jpp.inner_size * jpp.dt_size);

        jmp(loop_channel_blocks, T_NEAR);
    }

    L(loop_channel_tail); {
        cmp(reg_ch_work, 0);
        jle(loop_channel_end, T_NEAR);

        for (int i = 0; i < ur_inner; ++i) {
            load_vector(vreg_src(i), ptr[reg_src_ptr + i * simd_w * jpp.dt_size]);
            uni_vmaxps(vreg_max(i), vreg_max(i), vreg_src(i));
        }

        add(reg_src_ptr, jpp.inner_size * jpp.dt_size);

        dec(reg_ch_work);
        jmp(loop_channel_tail, T_NEAR);
    }

    L(loop_channel_end);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::simd_loop_exp(int ur_inner) {
    Label loop_channel_blocks;
    Label loop_channel_tail;
    Label loop_channel_end;

    for (int i = 0; i < ur_inner; ++i) {
        uni_vpxor(vreg_denom(i), vreg_denom(i), vreg_denom(i));
    }

    mov(reg_ch_work, reg_channels);

    mov(reg_src_ptr, reg_src_base_ptr);
    mov(reg_dst_ptr, reg_dst_base_ptr);

    L(loop_channel_blocks); {
        cmp(reg_ch_work, jpp.ur_channel);
        jl(loop_channel_tail, T_NEAR);

        for (int i = 0; i < ur_inner; ++i) {
            for (int c = 0; c < (int)jpp.ur_channel; ++c) {
                load_vector(vreg_src(i), ptr[reg_src_ptr + (i*simd_w + c*jpp.inner_size) * jpp.dt_size]);
                uni_vsubps(vreg_src(i),vreg_src(i), vreg_max(i));
                simd_expf(vreg_src(i));
                uni_vaddps(vreg_denom(i), vreg_denom(i), vreg_src(i));
                store_vector(ptr[reg_dst_ptr + (i*simd_w + c*jpp.inner_size) * jpp.dt_size], vreg_src(i));
            }
        }

        sub(reg_ch_work, jpp.ur_channel);
        add(reg_src_ptr, jpp.ur_channel * jpp.inner_size * jpp.dt_size);
        add(reg_dst_ptr, jpp.ur_channel * jpp.inner_size * jpp.dt_size);

        jmp(loop_channel_blocks, T_NEAR);
    }

    L(loop_channel_tail); {
        cmp(reg_ch_work, 0);
        jle(loop_channel_end, T_NEAR);

        for (int i = 0; i < ur_inner; ++i) {
            load_vector(vreg_src(i), ptr[reg_src_ptr + i * simd_w * jpp.dt_size]);
            uni_vsubps(vreg_src(i), vreg_src(i), vreg_max(i));
            simd_expf(vreg_src(i));
            uni_vaddps(vreg_denom(i), vreg_denom(i), vreg_src(i));
            store_vector(ptr[reg_dst_ptr + i * simd_w * jpp.dt_size], vreg_src(i));
        }

        add(reg_src_ptr, jpp.inner_size * jpp.dt_size);
        add(reg_dst_ptr, jpp.inner_size * jpp.dt_size);

        dec(reg_ch_work);
        jmp(loop_channel_tail, T_NEAR);
    }

    L(loop_channel_end);
}


template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::simd_loop_div(int ur_inner) {
    Label loop_channel_blocks;
    Label loop_channel_tail;
    Label loop_channel_end;

    for (int i = 0; i < ur_inner; ++i) {
        if (isa == sse41) {
            uni_vmovups(vmm_aux0, vmm_one);
            uni_vdivps(vmm_aux0, vmm_aux0, vreg_denom(i));
            uni_vmovups(vreg_denom(i), vmm_aux0);
        } else {
            uni_vdivps(vreg_denom(i), vmm_one, vreg_denom(i));
        }
    }

    mov(reg_ch_work, reg_channels);

    mov(reg_src_ptr, reg_src_base_ptr);
    mov(reg_dst_ptr, reg_dst_base_ptr);

    L(loop_channel_blocks); {
        cmp(reg_ch_work, jpp.ur_channel);
        jl(loop_channel_tail, T_NEAR);

        for (int i = 0; i < ur_inner; ++i) {
            for (int c = 0; c < (int)jpp.ur_channel; ++c) {
                load_vector(vreg_src(i), ptr[reg_dst_ptr + (i*simd_w + c*jpp.inner_size) * jpp.dt_size]);
                uni_vmulps(vreg_src(i), vreg_src(i), vreg_denom(i));
                store_vector(ptr[reg_dst_ptr + (i*simd_w + c*jpp.inner_size) * jpp.dt_size], vreg_src(i));
            }
        }

        sub(reg_ch_work, jpp.ur_channel);
        add(reg_src_ptr, jpp.ur_channel * jpp.inner_size * jpp.dt_size);
        add(reg_dst_ptr, jpp.ur_channel * jpp.inner_size * jpp.dt_size);

        jmp(loop_channel_blocks, T_NEAR);
    }

    L(loop_channel_tail); {
        cmp(reg_ch_work, 0);
        jle(loop_channel_end, T_NEAR);

        for (int i = 0; i < ur_inner; ++i) {
            load_vector(vreg_src(i), ptr[reg_dst_ptr + i * simd_w * jpp.dt_size]);
            uni_vmulps(vreg_src(i), vreg_src(i), vreg_denom(i));
            store_vector(ptr[reg_dst_ptr + i * simd_w * jpp.dt_size], vreg_src(i));
        }

        add(reg_src_ptr, jpp.inner_size * jpp.dt_size);
        add(reg_dst_ptr, jpp.inner_size * jpp.dt_size);

        dec(reg_ch_work);
        jmp(loop_channel_tail, T_NEAR);
    }

    L(loop_channel_end);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::scalar_loop_max() {
    Label loop_channel_tail;
    Label loop_channel_end;

    movups(xmm_max, xmm_float_min);
    mov(reg_src_ptr, reg_src_base_ptr);
    mov(reg_ch_work, reg_channels);

    L(loop_channel_tail); {
        cmp(reg_ch_work, 0);
        jle(loop_channel_end, T_NEAR);

        load_scalar(xmm_src, ptr[reg_src_ptr]);
        maxss(xmm_max, xmm_src);

        add(reg_src_ptr, jpp.inner_size * jpp.dt_size);

        dec(reg_ch_work);
        jmp(loop_channel_tail);
    }

    L(loop_channel_end);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::scalar_loop_exp() {
    Label loop_channel_tail;
    Label loop_channel_end;

    mov(reg_src_ptr, reg_src_base_ptr);
    mov(reg_dst_ptr, reg_dst_base_ptr);

    mov(reg_ch_work, reg_channels);

    pxor(xmm_denom, xmm_denom);

    L(loop_channel_tail); {
        cmp(reg_ch_work, 0);
        jle(loop_channel_end, T_NEAR);

        load_scalar(xmm_src, ptr[reg_src_ptr]);
        subss(xmm_src, xmm_max);
        scalar_expf(xmm_src);
        addss(xmm_denom, xmm_src);
        store_scalar(ptr[reg_dst_ptr], xmm_src);

        add(reg_src_ptr, jpp.inner_size * jpp.dt_size);
        add(reg_dst_ptr, jpp.inner_size * jpp.dt_size);

        dec(reg_ch_work);
        jmp(loop_channel_tail);
    }

    L(loop_channel_end);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::scalar_loop_div() {
    Label loop_channel_tail;
    Label loop_channel_end;

    mov(reg_src_ptr, reg_src_base_ptr);
    mov(reg_dst_ptr, reg_dst_base_ptr);
    mov(reg_ch_work, reg_channels);

    L(loop_channel_tail); {
        cmp(reg_ch_work, 0);
        jle(loop_channel_end, T_NEAR);

        load_scalar(xmm_src, ptr[reg_dst_ptr]);
        divss(xmm_src, xmm_denom);
        store_scalar(ptr[reg_dst_ptr], xmm_src);

        add(reg_src_ptr, jpp.inner_size * jpp.dt_size);
        add(reg_dst_ptr, jpp.inner_size * jpp.dt_size);

        dec(reg_ch_work);
        jmp(loop_channel_tail);
    }

    L(loop_channel_end);
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::dense_loop(int ou_block) {
    for (int ou = 0; ou < ou_block; ou++) {
        movups(xmm_max, xmm_float_min);
        for (int ch = 0; ch < (int)jpp.channels; ch++) {
            load_scalar(xmm_src, ptr[reg_src_base_ptr + (ou * jpp.channels + ch) * jpp.dt_size]);
            maxss(xmm_max, xmm_src);
        }

        for (int ch = 0; ch < (int)jpp.channels; ch++) {
            load_scalar(xmm_src, ptr[reg_src_base_ptr + (ou * jpp.channels + ch) * jpp.dt_size]);
            subss(xmm_src, xmm_max);
            store_scalar(ptr[reg_dst_base_ptr + (ou * jpp.channels + ch) * jpp.dt_size], xmm_src);
        }
    }

    int full_work = ou_block * (int)jpp.channels;
    int i = 0;
    for (; i <= full_work - simd_w; i += simd_w) {
        load_vector(vreg_src(0), ptr[reg_dst_base_ptr + i * jpp.dt_size]);
        simd_expf(vreg_src(0));
        store_vector(ptr[reg_dst_base_ptr + i * jpp.dt_size], vreg_src(0));
    }

    for (; i < full_work; i++) {
        load_scalar(xmm_src, ptr[reg_dst_base_ptr + i * jpp.dt_size]);
        scalar_expf(xmm_src);
        store_scalar(ptr[reg_dst_base_ptr + i * jpp.dt_size], xmm_src);
    }

    for (int ou = 0; ou < ou_block; ou++) {
        pxor(xmm_denom, xmm_denom);
        for (int ch = 0; ch < (int)jpp.channels; ch++) {
            load_scalar(xmm_src, ptr[reg_dst_base_ptr + (ou * jpp.channels + ch) * jpp.dt_size]);
            addss(xmm_denom, xmm_src);
        }

        movss(xmm_one, ptr[imm_addr64 + 0 * vlen]);
        divss(xmm_one, xmm_denom);
        movss(xmm_denom, xmm_one);
        for (int ch = 0; ch < (int)jpp.channels; ch++) {
            load_scalar(xmm_src, ptr[reg_dst_base_ptr + (ou * jpp.channels + ch) * jpp.dt_size]);
            mulss(xmm_src, xmm_denom);
            store_scalar(ptr[reg_dst_base_ptr + (ou * jpp.channels + ch) * jpp.dt_size], xmm_src);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::generate() {
    this->preamble();
    if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();

    if (jpp.inner_size == 1) {
        this->generate_dense();
    } else {
        mov(reg_src_base_ptr, ptr[abi_param1 + GET_OFF(src)]);
        mov(reg_dst_base_ptr, ptr[abi_param1 + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[abi_param1 + GET_OFF(work)]);
        mov(reg_channels, ptr[abi_param1 + GET_OFF(channels)]);

        mov(reg_min, float2int(-FLT_MAX));
        movq(xmm_float_min, reg_min);

        mov(imm_addr64, jit_uni_fork_softmax_kernel_f32<isa>::l_table);
        uni_vmovups(vmm_one, ptr[imm_addr64 + 0 * vlen]);

        cmp(reg_work_amount, jpp.ur_inner * simd_w);
        jl(loop_simd, T_NEAR);

        L(loop_simd_unroll);
        {
            simd_loop_max(jpp.ur_inner);
            simd_loop_exp(jpp.ur_inner);
            simd_loop_div(jpp.ur_inner);

            add(reg_src_base_ptr, jpp.ur_inner * simd_w * jpp.dt_size);
            add(reg_dst_base_ptr, jpp.ur_inner * simd_w * jpp.dt_size);

            sub(reg_work_amount, jpp.ur_inner * simd_w);
            cmp(reg_work_amount, jpp.ur_inner * simd_w);
            jge(loop_simd_unroll, T_NEAR);
        }

        L(loop_simd);
        {
            cmp(reg_work_amount, simd_w);
            jl(loop_scalar, T_NEAR);

            simd_loop_max(1);
            simd_loop_exp(1);
            simd_loop_div(1);

            add(reg_src_base_ptr, simd_w * jpp.dt_size);
            add(reg_dst_base_ptr, simd_w * jpp.dt_size);

            sub(reg_work_amount, simd_w);
            jmp(loop_simd, T_NEAR);
        }

        L(loop_scalar);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end, T_NEAR);

            scalar_loop_max();
            scalar_loop_exp();
            scalar_loop_div();

            add(reg_src_base_ptr, jpp.dt_size);
            add(reg_dst_base_ptr, jpp.dt_size);

            dec(reg_work_amount);
            jmp(loop_scalar, T_NEAR);
        }

        L(loop_end);

        this->postamble();

        prepare_table();
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_softmax_kernel_f32<isa>::generate_dense() {
    mov(reg_src_base_ptr, ptr[abi_param1 + GET_OFF(src)]);
    mov(reg_dst_base_ptr, ptr[abi_param1 + GET_OFF(dst)]);
    mov(reg_work_amount, ptr[abi_param1 + GET_OFF(work)]);

    mov(reg_min, float2int(-FLT_MAX));
    movq(xmm_float_min, reg_min);

    mov(imm_addr64, jit_uni_fork_softmax_kernel_f32<isa>::l_table);
    uni_vmovups(vmm_one, ptr[imm_addr64 + 0 * vlen]);

    int outer_tail = jpp.outer_size % jpp.outer_block;
    Label ou_loop_tail_label;
    Label ou_loop_tail_1_label;
    Label ou_loop_exit_label;

    cmp(reg_work_amount, jpp.outer_block);
    jne(ou_loop_tail_label, T_NEAR);

    dense_loop(jpp.outer_block);

    jmp(ou_loop_exit_label, T_NEAR);

    L(ou_loop_tail_label);
    cmp(reg_work_amount, outer_tail);
    jne(ou_loop_tail_1_label, T_NEAR);

    dense_loop(outer_tail);

    jmp(ou_loop_exit_label, T_NEAR);

    L(ou_loop_tail_1_label); {
        cmp(reg_work_amount, 1);
        jl(ou_loop_exit_label, T_NEAR);

        dense_loop(1);

        add(reg_src_base_ptr, jpp.dt_size * jpp.channels);
        add(reg_dst_base_ptr, jpp.dt_size * jpp.channels);
        dec(reg_work_amount);

        jmp(ou_loop_tail_1_label, T_NEAR);
    }

    L(ou_loop_exit_label);

    this->postamble();

    prepare_table();
}

template struct jit_uni_fork_softmax_kernel_f32<sse41>;
template struct jit_uni_fork_softmax_kernel_f32<avx2>;
template struct jit_uni_fork_softmax_kernel_f32<avx512_common>;

}
}
}
}
