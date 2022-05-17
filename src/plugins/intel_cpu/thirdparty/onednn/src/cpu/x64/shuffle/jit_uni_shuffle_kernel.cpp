/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <cassert>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/shuffle/jit_uni_shuffle_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_shuffle_call_s, field)

static size_t get_padding_size(const jit_shuffle_conf_t &conf) {
    const auto padding_tail_size = conf.c % conf.blk_size;
    return (padding_tail_size) ? conf.blk_size - padding_tail_size : 0;
}

template <cpu_isa_t isa>
jit_uni_shuffle_kernel_t<isa>::jit_uni_shuffle_kernel_t(
        const jit_shuffle_conf_t conf)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
    , conf_(conf)
    , padding_size_(get_padding_size(conf)) {}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::prepare_mask() {}

template <>
void jit_uni_shuffle_kernel_t<avx512_common>::prepare_mask() {
    const size_t tail_mask = (1ULL << conf_.simd_tail) - 1ULL;
    const Reg64 &reg_tail = reg_tmp_;
    mov(reg_tail.cvt32(), tail_mask);
    kmovw(k_tail_mask_, reg_tail.cvt32());
}

template <>
void jit_uni_shuffle_kernel_t<avx>::prepare_mask() {
    static constexpr uint32_t mask[16]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0, 0};
    mov(reg_tmp_, reinterpret_cast<size_t>(&mask[8 - conf_.simd_tail]));
    vmovups(vmm_tail_mask_, ptr[reg_tmp_]);
}

template <>
void jit_uni_shuffle_kernel_t<avx512_common>::emu_gather_data(
        const Reg64 &reg_src_addr, const int indices_idx, const int data_idx,
        const bool is_tail) {
    assert(conf_.data_type == data_type::bf16);

    const Xmm xmm_tmp = Xmm(vmm_full_mask_.getIdx());
    const Xmm xmm_dst = Xmm(vmm_tmp_.getIdx());

    xor_(reg_tmp_, reg_tmp_);
    mov(reg_tmp1_, reg_src_addr);

    constexpr unsigned xmm_size_elem = 8; //bf16
    constexpr unsigned xmm_size_elem_half = xmm_size_elem / 2;

    const unsigned number_of_xmms = is_tail
            ? utils::div_up(conf_.simd_tail, xmm_size_elem)
            : utils::div_up(conf_.simd_w, xmm_size_elem);

    for (unsigned i = 0; i < number_of_xmms; i++) {
        const unsigned number_of_xmm_halfs = is_tail && i == number_of_xmms - 1
                ? utils::div_up(
                        conf_.simd_tail, xmm_size_elem_half + i * xmm_size_elem)
                : 2;

        for (unsigned j = 0; j < number_of_xmm_halfs; j++) {
            const unsigned rem = conf_.simd_tail % xmm_size_elem_half;
            const unsigned number_of_values_to_load = is_tail
                            && i == number_of_xmms - 1
                            && j == number_of_xmm_halfs - 1 && rem
                    ? rem
                    : xmm_size_elem_half;

            vextractf32x4(xmm_tmp, Zmm(indices_idx), j + i * 2);
            for (unsigned k = 0; k < number_of_values_to_load; k++) {
                vpextrd(reg_tmp_.cvt32(), xmm_tmp, k + j * xmm_size_elem_half);
                add(reg_src_addr, reg_tmp_);
                vpinsrw(xmm_dst, xmm_dst, ptr[reg_src_addr],
                        k + j * xmm_size_elem_half);
                mov(reg_src_addr, reg_tmp1_);
            }
        }
        vinsertf128(Ymm(data_idx), Ymm(data_idx), xmm_dst, i);
    }
}

template <>
void jit_uni_shuffle_kernel_t<avx>::emu_gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    const Xmm xmm_tmp = Xmm(vmm_full_mask_.getIdx());
    const Xmm xmm_dst = Xmm(vmm_tmp_.getIdx());

    xor_(reg_tmp_, reg_tmp_);
    mov(reg_tmp1_, reg_src_addr);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_xmms = is_tail
            ? utils::div_up(conf_.simd_tail, xmm_size_elem)
            : utils::div_up(conf_.simd_w, xmm_size_elem);
    for (unsigned i = 0; i < number_of_xmms; i++) {
        vextractf128(xmm_tmp, Ymm(indices_idx), i);

        const unsigned number_of_values_to_load = i == number_of_xmms - 1
                        && is_tail && conf_.simd_tail % xmm_size_elem != 0
                ? conf_.simd_tail % xmm_size_elem
                : xmm_size_elem;
        for (unsigned j = 0; j < number_of_values_to_load; j++) {
            vpextrd(reg_tmp_.cvt32(), xmm_tmp, j);
            add(reg_src_addr, reg_tmp_);
            vpinsrd(xmm_dst, xmm_dst, ptr[reg_src_addr], j);
            mov(reg_src_addr, reg_tmp1_);
        }

        vinsertf128(Ymm(data_idx), Ymm(data_idx), xmm_dst, i);
    }
}

template <>
void jit_uni_shuffle_kernel_t<sse41>::emu_gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    xor_(reg_tmp_, reg_tmp_);
    mov(reg_tmp1_, reg_src_addr);

    constexpr unsigned xmm_size_elem = 4;

    const unsigned number_of_values_to_load
            = is_tail ? conf_.simd_tail : xmm_size_elem;
    for (unsigned j = 0; j < number_of_values_to_load; j++) {
        pextrd(reg_tmp_.cvt32(), Xmm(indices_idx), j);
        add(reg_src_addr, reg_tmp_);
        pinsrd(Xmm(data_idx), ptr[reg_src_addr], j);
        mov(reg_src_addr, reg_tmp1_);
    }
}

template <>
void jit_uni_shuffle_kernel_t<avx512_common>::gather_data(
        const Reg64 &reg_src_addr, const int indices_idx, const int data_idx,
        const bool is_tail) {
    if (conf_.dt_size == sizeof(float)) {
        const Opmask &mask = is_tail ? k_tail_mask_ : k_full_mask_;
        if (!is_tail) {
            // Have to set the all bits to 1 gather full
            // register. It is needed after each gather, because
            // vgatherdps zeros the mask if successful
            mov(reg_tmp_.cvt32(), (1ULL << conf_.simd_w) - 1ULL);
            kmovw(k_full_mask_, reg_tmp_.cvt32());
        }
        vgatherdps(Vmm(data_idx) | mask, ptr[reg_src_addr + Vmm(indices_idx)]);
        // Have to restore tail processing mask after gather because mask
        // was zeroed after vgatherdps.
        if (is_tail) prepare_mask();
    } else {
        emu_gather_data(reg_src_addr, indices_idx, data_idx, is_tail);
    }
}

template <>
void jit_uni_shuffle_kernel_t<avx>::gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    if (conf_.isa == avx2 && conf_.dt_size == sizeof(float)) {
        const Vmm &mask = is_tail ? vmm_tail_mask_ : vmm_full_mask_;
        if (!is_tail) {
            // Have to set the all bits to 1 gather full
            // register. It is needed after each gather, because
            // vgatherdps zeros the mask if successful
            if (conf_.data_type == data_type::s32)
                vpcmpeqw(vmm_full_mask_, vmm_full_mask_, vmm_full_mask_);
            else
                vcmpps(vmm_full_mask_, vmm_full_mask_, vmm_full_mask_,
                        _cmp_eq_oq);
        }
        if (conf_.data_type == data_type::s32)
            vpgatherdd(
                    Vmm(data_idx), ptr[reg_src_addr + Vmm(indices_idx)], mask);
        else
            vgatherdps(
                    Vmm(data_idx), ptr[reg_src_addr + Vmm(indices_idx)], mask);
        // Have to restore tail processing mask after gather because mask
        // was zeroed after vgatherdps.
        if (is_tail) prepare_mask();
    } else {
        emu_gather_data(reg_src_addr, indices_idx, data_idx, is_tail);
    }
}

template <>
void jit_uni_shuffle_kernel_t<sse41>::gather_data(const Reg64 &reg_src_addr,
        const int indices_idx, const int data_idx, const bool is_tail) {
    emu_gather_data(reg_src_addr, indices_idx, data_idx, is_tail);
}

template <>
void jit_uni_shuffle_kernel_t<avx512_common>::store_data(const int data_idx,
        const Reg64 &reg_dst_addr, const int offset, const bool is_tail) {
    const auto extend_for_padding
            = is_tail && padding_size_ + conf_.simd_tail >= conf_.simd_w;
    if (conf_.data_type == data_type::bf16) {
        const Ymm to_store_data = Ymm(data_idx);
        const Ymm ymm_tmp = Ymm(vmm_tmp_.getIdx());

        if (extend_for_padding) {
            vmovdqu16(ymm_tmp | k_tail_mask_ | T_z, to_store_data);
            vmovups(ptr[reg_dst_addr + offset], ymm_tmp);
        } else {
            if (is_tail)
                vmovdqu16(ptr[reg_dst_addr + offset] | k_tail_mask_,
                        to_store_data);
            else
                vmovups(ptr[reg_dst_addr + offset], to_store_data);
        }
    } else {
        if (extend_for_padding) {
            vmovups(vmm_tmp_ | k_tail_mask_ | T_z, Vmm(data_idx));
            vmovups(ptr[reg_dst_addr + offset], vmm_tmp_);
        } else {
            if (is_tail)
                vmovups(ptr[reg_dst_addr + offset] | k_tail_mask_,
                        Vmm(data_idx));
            else
                vmovups(ptr[reg_dst_addr + offset], Vmm(data_idx));
        }
    }
    append_zero_padding(reg_dst_, extend_for_padding);
}

template <>
void jit_uni_shuffle_kernel_t<avx>::store_data(const int data_idx,
        const Reg64 &reg_dst_addr, const int offset, const bool is_tail) {
    const auto extend_for_padding
            = is_tail && padding_size_ + conf_.simd_tail >= conf_.simd_w;

    if (extend_for_padding) {
        uni_vxorps(vmm_tmp_, vmm_tmp_, vmm_tmp_);
        uni_vblendvps(vmm_tmp_, vmm_tmp_, Vmm(data_idx), vmm_tail_mask_);
        vmovups(ptr[reg_dst_addr + offset], vmm_tmp_);
    } else {
        if (is_tail)
            vmaskmovps(
                    ptr[reg_dst_addr + offset], vmm_tail_mask_, Vmm(data_idx));
        else
            vmovups(ptr[reg_dst_addr + offset], Vmm(data_idx));
    }
    append_zero_padding(reg_dst_, extend_for_padding);
}

template <>
void jit_uni_shuffle_kernel_t<sse41>::store_data(const int data_idx,
        const Reg64 &reg_dst_addr, const int offset, const bool is_tail) {
    if (is_tail)
        for (unsigned i = 0; i < conf_.simd_tail; i++) {
            pextrd(ptr[reg_dst_addr + offset + i * conf_.dt_size],
                    Xmm(data_idx), i);
        }
    else
        movups(ptr[reg_dst_addr + offset], Vmm(data_idx));

    append_zero_padding(reg_dst_, false);
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::shuffle_blocked_format() {
    const Reg64 &reg_sp = reg_tmp2_;
    const Reg64 &reg_cb = reg_tmp3_;
    const Reg64 &reg_cb_loop_size = reg_tmp4_;
    const Reg64 &reg_blk_tail = reg_tmp5_;
    const Reg64 &reg_src_save = reg_tmp6_;
    const int simd_in_blk = conf_.blk_size / conf_.simd_w;
    const int simd_in_tail_blk
            = utils::div_up(conf_.c % conf_.blk_size, conf_.simd_w);
    const Vmm vmm_tmp[4] = {Vmm(5), Vmm(6), Vmm(7), Vmm(8)};

    auto load_indices = ([&](bool is_blk_tail) {
        const int simd_to_process
                = is_blk_tail ? simd_in_tail_blk : simd_in_blk;
        for (int i = 0; i < simd_to_process; ++i)
            uni_vmovdqu(vmm_tmp[i],
                    ptr[reg_indices_
                            + i * conf_.simd_w * conf_.el_size_of_indices]);
    });

    auto shuffle = ([&](bool is_blk_tail) {
        const int simd_to_process
                = is_blk_tail ? simd_in_tail_blk : simd_in_blk;
        for (int i = 0; i < simd_to_process; ++i) {
            const bool simd_tail_condition = is_blk_tail && conf_.simd_tail > 0
                    && i == simd_to_process - 1;
            uni_vmovups(vmm_indices_, vmm_tmp[i]);
            gather_data(reg_src_, vmm_indices_.getIdx(), vmm_src_.getIdx(),
                    simd_tail_condition);

            store_data(vmm_src_.getIdx(), reg_dst_,
                    i * conf_.simd_w * conf_.dt_size, simd_tail_condition);
        }
    });

    mov(reg_cb_loop_size, ptr[reg_param + GET_OFF(cb_loop_size)]);

    Label sp_loop_begin, sp_loop_end;
    Label sp_tail_loop_begin, sp_tail_loop_end;
    Label cb_loop_begin, cb_loop_end;
    Label simd_loop_begin, simd_loop_end;
    Label blk_tail_loop_begin, blk_tail_loop_end;
    Label blk_tail_check_end;
    Label no_tail;

    xor_(reg_blk_tail, reg_blk_tail);

    cmp(reg_cb_loop_size, conf_.blk_size);
    je(no_tail, T_NEAR);

    mov(reg_blk_tail, reg_cb_loop_size);
    xor_(reg_cb_loop_size, reg_cb_loop_size);

    L(no_tail);

    xor_(reg_cb, reg_cb);
    L(cb_loop_begin);
    {
        cmp(reg_cb, reg_cb_loop_size);
        jge(cb_loop_end, T_NEAR);

        load_indices(false);

        mov(reg_src_save, reg_src_);

        xor_(reg_sp, reg_sp);
        L(sp_loop_begin);
        {
            cmp(reg_sp, conf_.sp_split_size);
            jge(sp_loop_end, T_NEAR);

            shuffle(false);

            inc(reg_sp);
            add(reg_src_, conf_.blk_size * conf_.dt_size);
            add(reg_dst_, conf_.blk_size * conf_.dt_size);

            jmp(sp_loop_begin);
        }
        L(sp_loop_end);

        mov(reg_src_, reg_src_save);

        add(reg_cb, conf_.blk_size);
        add(reg_dst_,
                conf_.blk_size * (conf_.sp - conf_.sp_split_size)
                        * conf_.dt_size);
        add(reg_indices_, conf_.blk_size * conf_.el_size_of_indices);

        jmp(cb_loop_begin);
    }
    L(cb_loop_end);

    cmp(reg_blk_tail, 0);
    je(blk_tail_check_end, T_NEAR);

    load_indices(true);

    xor_(reg_sp, reg_sp);
    L(sp_tail_loop_begin);
    {
        cmp(reg_sp, conf_.sp_split_size);
        jge(sp_tail_loop_end, T_NEAR);

        shuffle(true);

        inc(reg_sp);
        add(reg_src_, conf_.blk_size * conf_.dt_size);
        add(reg_dst_, conf_.blk_size * conf_.dt_size);

        jmp(sp_tail_loop_begin);
    }
    L(sp_tail_loop_end);

    L(blk_tail_check_end);
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::append_zero_padding(
        const Reg64 &reg_dst_addr, const bool extend_for_padding) {

    static constexpr size_t reg64_size = 8;
    const size_t simd_w_byte = conf_.simd_w * sizeof(float);

    if (!padding_size_) return;

    const auto padding_start
            = (extend_for_padding) ? conf_.simd_w : conf_.c % conf_.blk_size;
    const auto padding_end = (extend_for_padding)
            ? padding_size_ - (conf_.simd_w - conf_.simd_tail)
            : padding_size_;
    const auto off_start = padding_start * conf_.dt_size;
    const auto padding_to_add = padding_end * conf_.dt_size;

    if (!padding_to_add) return;

    Label end;
    unsigned int off = 0;

    cmp(reg_padded_block, 0);
    je(end, T_NEAR);

    if (simd_w_byte <= padding_to_add) {
        uni_vxorps(vmm_zero_, vmm_zero_, vmm_zero_);
        for (; off + simd_w_byte < padding_to_add; off += simd_w_byte)
            uni_vmovups(ptr[reg_dst_addr + off_start + off], vmm_zero_);
    }

    if (off != padding_to_add) {
        xor_(reg_tmp_, reg_tmp_);
        for (; off + reg64_size < padding_to_add; off += reg64_size)
            mov(ptr[reg_dst_addr + off_start + off], reg_tmp_);
        for (; off < padding_to_add; off++)
            mov(ptr[reg_dst_addr + off_start + off], Reg8(reg_tmp_.getIdx()));
    }

    L(end);
}

template <cpu_isa_t isa>
void jit_uni_shuffle_kernel_t<isa>::generate() {
    preamble();

#if defined(_WIN32)
    // Always mimic the Unix ABI
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif

    if (conf_.isa == avx2) {
        // Sometimes the values in the register can be nan at the
        // beginning of the kernel, then using vcmpps(vmm, vmm, vmm)
        // will not set all bits to 1 for that value, instead
        // this instruction will zeroed this value. At the beginning,
        // it is worth to zeroing this register to be sure, that vcmpps
        // will work properly.
        uni_vxorps(vmm_full_mask_, vmm_full_mask_, vmm_full_mask_);
    }

    if (conf_.simd_tail > 0) prepare_mask();

    mov(reg_indices_, ptr[reg_param + GET_OFF(input_off_ptr)]);

    mov(reg_src_, ptr[reg_param + GET_OFF(src)]);
    mov(reg_dst_, ptr[reg_param + GET_OFF(dst)]);
    mov(reg_padded_block, ptr[reg_param + GET_OFF(is_padded_block)]);

    shuffle_blocked_format();

    postamble();
}

template struct jit_uni_shuffle_kernel_t<sse41>;
template struct jit_uni_shuffle_kernel_t<avx>;
template struct jit_uni_shuffle_kernel_t<avx512_common>;

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
