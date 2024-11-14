// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_load_store_emitters.hpp"
#include "utils/bfloat16.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
using namespace Xbyak::util;

// This macro is to enable instruction keep values in source vector unchanged after execution.
// An auxiliary vector reg(data_reg_new) is used as destination vector for source pollution instructions,
// After updated, processed with new vector and no more need to update as source is preserved.
// e.g. with STORE_KEEP_SOURCE(vextractf128, xmm, Xmm(aux_src_idx), ymm, 1);
//      if ymm is already updated, h->vextractf128(xmm, ymm, 1) is used, which change ymm values as xmm and ymm have the same index.
//      if ymm is not updated, h->vextractf128(Xmm(aux_src_idx), ymm, 1) is used, which keep ymm values unchanged as destination is another vector reg.
#define STORE_KEEP_SOURCE(instruction, data_reg, data_reg_new, ...) \
    if (data_reg_updated) { \
        h->instruction(data_reg, __VA_ARGS__); \
    } else { \
        h->instruction(data_reg_new, __VA_ARGS__); \
        data_idx = aux_src_idx; \
        xmm = Xbyak::Xmm(data_idx); \
        ymm = Xbyak::Ymm(data_idx); \
        zmm = Xbyak::Zmm(data_idx); \
        vmm = Vmm(data_idx); \
        data_reg_updated = true; \
    }

namespace ov {
namespace intel_cpu {

namespace {
// heuristic threshold number by byte between mask load and emulation with several simple partial load
constexpr int threshold_for_mask_emu_load = 14;
// heuristic threshold number by byte between mask store and emulation with several simple partial store
constexpr int threshold_for_mask_emu_store = 6;
}   // namespace

size_t load_emitter_params::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, std::string("jit_load_emitter"));
    seed = hash_combine(seed, src_prc_.hash());
    seed = hash_combine(seed, dst_prc_.hash());
    seed = hash_combine(seed, load_num_);
    seed = hash_combine(seed, is_fill_);
    seed = hash_combine(seed, fill_value_);
    return seed;
}

size_t store_emitter_params::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, std::string("jit_store_emitter"));
    seed = hash_combine(seed, src_prc_.hash());
    seed = hash_combine(seed, dst_prc_.hash());
    seed = hash_combine(seed, store_num_);
    return seed;
}

static int get_aux_regs_as_temp(const int elem_count, const int data_size, bool is_pure_move, bool is_store_as_real16,
                                const int avx512_threshold_for_mask = 0, const bool is_fill = false) {
    if (mayiuse(cpu::x64::avx512_core) && is_fill)
        return 1;
    // for pure move, there are direct no-mask instructions to move on full xmm/ymm/zmm, so aux_gpr is not needed.
    // for move+convert:
    // there are direct no-mask instructions to load i8/u8/i16/u16/bf16/fp16 to full xmm/ymm/zmm as f32/i32, so aux_gpr is not needed.
    // there are direct no-mask instructions to store i32 on full xmm/ymm/zmm to i8/u8/i16/u16, so aux_gpr is not needed.
    // store f32 on full xmm/ymm/zmm to bf16/fp16, need convert to bf16/fp16 on vmm, then store vmm to memory, use store_dword_to_word/byte_base condition.
    // store_num == 16, vector: 16 * f32 -> 16 * bf16 -> ymm(256bit) -> store
    // store_num == 8,  vector:  8 * f32 ->  8 * bf16 -> xmm(128bit)  -> store
    // store_num == 4,  vector:  4 * f32 ->  4 * bf16 ->       64bit  -> masked instruction with aux_gpr needed
    // f32<->i32 is on full vmm, so aux_gpr is not needed.
    const int byte_size = elem_count * data_size;
    if ((is_pure_move && one_of(byte_size, 16, 32, 64)) || (!is_pure_move && one_of(elem_count, 4, 8, 16) && !is_store_as_real16))
        return 0;
    if ((mayiuse(cpu::x64::avx512_core) && (byte_size > avx512_threshold_for_mask)) || (one_of(byte_size % 16, 1, 2, 3)))
        return 1;
    return 0;
}

/// LOAD ///
jit_load_emitter::jit_load_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                   ov::element::Type src_prc, ov::element::Type dst_prc, int load_num, ov::element::Type exec_prc,
                                   bool is_fill, std::string fill_value, emitter_in_out_map in_out_type)
: jit_emitter(host, host_isa, exec_prc, in_out_type), name_("unknown"), load_num_(load_num), src_prc_(src_prc),
    dst_prc_(dst_prc), is_fill_(is_fill), fill_value_(fill_value) {
    prepare_table();
    load_size_ = load_num * src_prc.size();
    v_len_elt_ = get_vec_length() / exec_prc.size();
}

size_t jit_load_emitter::get_inputs_num() const { return 1; }

size_t jit_load_emitter::aux_gprs_count() const {
    // 0 for temp reg for mask load in avx512 if needed
    const auto is_pure_load = (src_prc_ == dst_prc_) ||
                                (one_of(src_prc_, ov::element::f32, ov::element::i32) &&
                                 one_of(dst_prc_, ov::element::f32, ov::element::i32));
    int count = get_aux_regs_as_temp(load_num_, static_cast<int>(src_prc_.size()), is_pure_load, false, threshold_for_mask_emu_load, is_fill_);

    // 1 for table address
    if (is_fill_)
        count++;

    return count;
}

void jit_load_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    // offset in load emitter is the offset of src gpr register, should be parsed from in_idxs.
    const int offset = in_idxs.size() == 2 ? in_idxs[1] : 0;
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(Reg64(in_idxs[0]), static_cast<int>(out_idxs[0]), offset);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(Reg64(in_idxs[0]), static_cast<int>(out_idxs[0]), offset);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        emit_isa<cpu::x64::avx512_core>(Reg64(in_idxs[0]), static_cast<int>(out_idxs[0]), offset);
    } else {
        OV_CPU_JIT_EMITTER_THROW("is performed on unsupported isa(at least x64::sse41).");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_load_emitter::emit_isa(const Xbyak::Reg64 &reg_src, const int out_vec_idx, const int offset) const {
    bool matched_prc = (dst_prc_ == src_prc_) || (dst_prc_ == ov::element::f32) || (dst_prc_ == ov::element::i32);
    if (!matched_prc) {
        OV_CPU_JIT_EMITTER_THROW("only support output precision of FP32 or I32 or the same precision as input.");
    }
    if (load_num_ > static_cast<int>((get_vec_length() / dst_prc_.size()))) {
        OV_CPU_JIT_EMITTER_THROW("have unexpected number of elements to load.");
    }

    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;

    // pure load
    if (src_prc_ == dst_prc_) {
        load_bytes<Vmm>(Vmm(out_vec_idx), reg_src, offset, load_size_);
    } else {
    // "pure load" + convert. dst_prc must be FP32 or I32.
        switch (src_prc_) {
            case ov::element::f32:
            case ov::element::i32:
                load_bytes<Vmm>(Vmm(out_vec_idx), reg_src, offset, load_size_);
                break;
            case ov::element::i8:
                load_bytes_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset, true, load_size_);
                break;
            case ov::element::u8:
                load_bytes_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset, false, load_size_);
                break;
            case ov::element::i16:
            case ov::element::u16:
            case ov::element::bf16:
            case ov::element::f16:
                load_words_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset, src_prc_, load_size_);
                break;
            default:
                OV_CPU_JIT_EMITTER_THROW("has unsupported src precision to load.");
        }
    }

    // post convert between I32 and FP32
    if (src_prc_ != dst_prc_) {
        switch (dst_prc_) {
            case ov::element::f32:
                if (!src_prc_.is_real())
                    h->uni_vcvtdq2ps(Vmm(out_vec_idx), Vmm(out_vec_idx));
                break;
            case ov::element::i32:
                if (src_prc_.is_real()) {
                    h->uni_vcvtps2dq(Vmm(out_vec_idx), Vmm(out_vec_idx));
                }
                break;
            default:
                break;
        }
    }

    if (is_fill_) {
        int dword_num_loaded = (src_prc_ != dst_prc_) ? load_num_ : (load_size_ / sizeof(float));
        fill_with_default(Vmm(out_vec_idx), fill_value_, dword_num_loaded);
    }
}

/**
* load_bytes is the utility function to facilitate loading of
* load_size (0 <= load_size <= 64) many contiguous bytes into the Xmm/Ymm/Zmm
* register from the memory referenced by ptr[reg + offset] address.
*
* Functionally, invocation of load_bytes is equivalent to
* the following loop:
*
* for (int idx = 0; idx < load_size; ++idx)
*     vpinsrb(vmm, vmm, ptr[reg + offset + idx], idx);
*
*/
template <typename Vmm>
void jit_load_emitter::load_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, int load_size) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure data fits completely inside the Xmm/Ymm/Zmm register
    if (load_size < 0 || load_size > 64)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load in load_byte.");
    // check if proper number bytes fit inside the Xmm/Ymm register
    if (is_ymm && load_size > 32)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load to ymm in load_byte.");
    if (is_xmm && load_size > 16)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load to xmm in load_byte.");

    auto xmm = Xbyak::Xmm(vmm.getIdx());
    auto ymm = Xbyak::Ymm(vmm.getIdx());
    auto zmm = Xbyak::Zmm(vmm.getIdx());

    // addr(i) denotes the memory pointed by ptr[reg + offset + (i bytes)]
    const auto addr = [&](int bytes_offset) {
        return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
    };
    const auto word_addr = [&](int bytes_offset) {
        return word[reg + offset + bytes_offset * sizeof(int8_t)];
    };

    auto load_byte_base = [&]() {
        int start_bytes = 0;
        int bytes_to_load = load_size;

        bool has_ymm_block = false;
        if (bytes_to_load > 32) {
            // Prepare to insert to upper bits of zmm
            start_bytes += 32;
            bytes_to_load -= 32;
            has_ymm_block = true;
        }

        bool has_xmm_block = false;
        if (bytes_to_load > 16) {
            // Prepare to insert to upper bits of ymm
            start_bytes += 16;
            bytes_to_load -= 16;
            has_xmm_block = true;
        }

        // Cornerstone of partial load is combinaion of vpinsrb/w/d.
        // As vpinsrb/w/d will not only write(insert) values into vmm, but also read values in vmm to copy from to positions that not in imm mask,
        // this could introduce RAW false dependency(we actually do not care about values not in imm mask).
        // To eliminate this false dependency,
        // 1. For 1/2/3/4 bytes tails, replace vpinsrb/w/d with mov,shl etc instructions that don't read vmm.
        //    Besides eliminate RAW, these instructions have smaller latency, which also bring better perf, especially for small loop iteration case.
        // 2. For 8/16 bytes, use vmovq/vmovdqu instructions to load, which also don't read src vmm.
        // 3. For other size, insert vpxor before vpinsrb/w/d. vpxor and read vmm instructions in previous loop have WAR(write after read) relationship.
        //    CPU can identify this scenario and assign another physical vector register(register renameing) in next loop to eliminate RAW.
        if (!one_of(bytes_to_load, 0, 1, 2, 3, 4, 8, 16)) {
            h->uni_vpxor(vmm, vmm, vmm);
        }
        if (bytes_to_load >= 8 && bytes_to_load < 16)
            h->uni_vmovq(xmm, addr(start_bytes));
        else if (bytes_to_load == 16)
            h->uni_vmovdqu(xmm, addr(start_bytes));

        switch (bytes_to_load) {
            case 0: break;
            case 1:
                h->movzx(Reg32(aux_gpr_idxs[0]), addr(start_bytes));
                h->uni_vmovq(xmm, Reg64(aux_gpr_idxs[0]));
                break;
            case 2:
                h->movzx(Reg32(aux_gpr_idxs[0]), word_addr(start_bytes));
                h->uni_vmovq(xmm, Reg64(aux_gpr_idxs[0]));
                break;
            case 3:
                h->movzx(Reg32(aux_gpr_idxs[0]), addr(start_bytes + 2));
                h->shl(Reg32(aux_gpr_idxs[0]), 16);
                h->mov(Reg16(aux_gpr_idxs[0]), word_addr(start_bytes));
                h->uni_vmovq(xmm, Reg64(aux_gpr_idxs[0]));
                break;
            case 4: h->uni_vmovss(xmm, addr(start_bytes)); break;
            case 5:
                h->uni_vmovss(xmm, addr(start_bytes));
                h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 4), 4);
                break;
            case 6:
                h->uni_vmovss(xmm, addr(start_bytes));
                h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 4), 2);
                break;
            case 7:
                h->uni_vmovss(xmm, addr(start_bytes));
                h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 4), 2);
                h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 6), 6);
                break;
            case 8: break;
            case 9: h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 8), 8); break;
            case 10: h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 8), 4); break;
            case 11:
                h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 8), 4);
                h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 10), 10);
                break;
            case 12: h->uni_vpinsrd(xmm, xmm, addr(start_bytes + 8), 2); break;
            case 13:
                h->uni_vpinsrd(xmm, xmm, addr(start_bytes + 8), 2);
                h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 12), 12);
                break;
            case 14:
                h->uni_vpinsrd(xmm, xmm, addr(start_bytes + 8), 2);
                h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 12), 6);
                break;
            case 15:
                h->uni_vpinsrd(xmm, xmm, addr(start_bytes + 8), 2);
                h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 12), 6);
                h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 14), 14);
                break;
            case 16: break;
            default:
                OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load in load_byte.");
        }

        if (has_xmm_block) {
            h->vinsertf128(ymm, ymm, xmm, 1); // insert to upper bits of ymm
            if (has_ymm_block)
                h->vinsertf128(ymm, ymm, addr(32), 0); // insert to lower bits of ymm
            else
                h->vinsertf128(ymm, ymm, addr(0), 0); // insert to lower bits of ymm
        }

        if (has_ymm_block) {
            h->vinsertf64x4(zmm, zmm, ymm, 1); // insert to upper bits of zmm
            h->vinsertf64x4(zmm, zmm, addr(0), 0); // insert to lower bits of zmm
        }
    };

    switch (load_size) {
        case 64:
            h->uni_vmovdqu(zmm, addr(0));
            break;
        case 32:
            h->uni_vmovdqu(ymm, addr(0));
            break;
        case 16:
            h->uni_vmovdqu(xmm, addr(0));
            break;
        default: {
            if (mayiuse(cpu::x64::avx512_core) && load_size > threshold_for_mask_emu_load) {
                uint64_t mask = 1;
                mask = (mask << load_size) - mask;
                h->mov(Reg64(aux_gpr_idxs[0]), mask);
                h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
                h->vmovdqu8(zmm | k_mask | T_z, addr(0));
            } else {
                load_byte_base();
            }
            break;
        }
    }
}

/**
* load_bytes_to_dword_extension is the utility function to facilitate
* loading of load_size (0 <= load_size <= 16) many contiguous bytes in
* the xmm register from the memory referenced by ptr[reg + offset]
* address and then do signed/zero extension of those to double words.
*
* Functionally, invocation of load_bytes_to_dword_extension is equivalent
* to the following:
*
* for (int idx = 0; idx < load_size; ++idx)
*     vpinsrb(vmm, vmm, ptr[reg + offset + idx], idx);
* if (is_signed) vpmovsxbd(vmm, vmm); else vpmovzxbd(vmm, vmm);
*
* Valid values for the load_size variable are:
* [0..4] for XMM version of the function, i.e. 4 bytes -> 4 * 32 bit == 128 bit
* [0..8] for YMM version of the function. i.e. 8 bytes -> 8 * 32 bit == 256 bit
* [0..16] for ZMM version of the function. i.e. 16 bytes -> 16 * 32 bit == 512 bit
*/

template <typename Vmm>
void jit_load_emitter::load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_signed, int load_size) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure extended double words fit inside Zmm (32 * load_size <= 512)
    // For Ymm register, load capacity is halved (32 * load_size <= 256)
    // For Xmm register, load capacity is halved further (32 * load_size <= 128)
    if (load_size < 0 || load_size > 16)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load in load_bytes_to_dword_extension.");
    if (is_ymm && load_size > 8)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load to ymm in load_bytes_to_dword_extension.");
    if (is_xmm && load_size > 4)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load to xmm in load_bytes_to_dword_extension.");

    // For load_size == 4/8/16, do load/extension in one go
    switch (load_size) {
        case 16: {
            // full size of zmm
            const auto zmm = Xbyak::Zmm(vmm.getIdx());
            if (is_signed)
                h->uni_vpmovsxbd(zmm, ptr[reg + offset]);
            else
                h->uni_vpmovzxbd(zmm, ptr[reg + offset]);
            break;
        }
        case 8: {
            // full size of ymm or ymm_block of zmm
            const auto ymm = Xbyak::Ymm(vmm.getIdx());
            if (is_signed)
                h->uni_vpmovsxbd(ymm, ptr[reg + offset]);
            else
                h->uni_vpmovzxbd(ymm, ptr[reg + offset]);
            break;
        }
        case 4: {
            // full size of xmm or xmm_block of ymm/zmm
            const auto xmm = Xbyak::Xmm(vmm.getIdx());
            if (is_signed)
                h->uni_vpmovsxbd(xmm, ptr[reg + offset]);
            else
                h->uni_vpmovzxbd(xmm, ptr[reg + offset]);
            break;
        }
        default: {
            if (is_zmm && load_size > threshold_for_mask_emu_load) {
                unsigned int mask = 1;
                mask = (mask << load_size) - mask;
                h->mov(Reg32(aux_gpr_idxs[0]), mask);
                h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
                if (is_signed)
                    h->uni_vpmovsxbd(vmm | k_mask | T_z, ptr[reg + offset]);
                else
                    h->uni_vpmovzxbd(vmm | k_mask | T_z, ptr[reg + offset]);
            } else {
                const auto xmm = Xbyak::Xmm(vmm.getIdx());
                load_bytes(xmm, reg, offset, load_size);
                if (is_signed)
                    h->uni_vpmovsxbd(vmm, xmm);
                else
                    h->uni_vpmovzxbd(vmm, xmm);
            }
            break;
        }
    }
}

/**
* load_words_to_dword_extension is the utility function to facilitate
* loading of load_size (0 <= load_size <= 32) byte many contiguous words(num == load_size / 2)
* in the Vmm register from the memory referenced by ptr[reg + offset]
* address and then do signed/zero extension of those to double words.
*
* Functionally, invocation of load_words_to_dword_extension is equivalent
* to the following extended pseudo code:
*
* for (int idx = 0; idx < load_size / 2; ++idx)
*     vpinsrw(vmm, vmm, ptr[reg + offset + 2 * idx], idx);
* if (is_signed) vpmovsxwd(vmm, vmm); else vpmovzxwd(vmm, vmm);
*
* Valid values for the load_size variable are:
* [0..8] for XMM version of the function.  i.e. 4 words -> 4 * 32 bit == 128 bit
* [0..16] for YMM version of the function. i.e. 8 words -> 8 * 32 bit == 256 bit
* [0.. 32] for ZMM version of the function. i.e. 16 words -> 16 * 32 bit == 512 bit
*/
template <typename Vmm>
void jit_load_emitter::load_words_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, ov::element::Type prc, int load_size) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    bool is_bf16 = (prc == ov::element::bf16);
    bool is_f16 = (prc == ov::element::f16);
    bool is_signed = prc.is_signed();

    if (is_f16 && !mayiuse(cpu::x64::avx2))
        OV_CPU_JIT_EMITTER_THROW("only support fp16 on platform with avx2 or above.");

    // Ensure extended double words fit inside Zmm (32/2(num) * 32 <= 512)
    // For Ymm register, load capacity is halved (16/2(num) * 32 <= 128)
    // For Xmm register, load capacity is halved again (8/2(num) * 32 <= 128)
    if (load_size < 0 || load_size > 32)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load in load_words_to_dword_extension.");
    if (is_ymm && load_size > 16)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load to ymm in load_words_to_dword_extension.");
    if (is_xmm && load_size > 8)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to load to xmm in load_words_to_dword_extension.");

    auto xmm = Xbyak::Xmm(vmm.getIdx());
    auto ymm = Xbyak::Ymm(vmm.getIdx());
    auto zmm = Xbyak::Zmm(vmm.getIdx());

    // For load_size == 32/16/8, do load/extension in one go
    // including xmm/ymm tail block for ymm/zmm, so explicite xmm/ymm/zmm
    switch (load_size) {
        case 32: {
            if (is_bf16) {
                h->uni_vpmovzxwd(zmm, ptr[reg + offset]);
                h->uni_vpslld(zmm, zmm, 16);
            } else if (is_f16) {
                h->vcvtph2ps(zmm, ptr[reg + offset]);
            } else {
                if (is_signed)
                    h->uni_vpmovsxwd(zmm, ptr[reg + offset]);
                else
                    h->uni_vpmovzxwd(zmm, ptr[reg + offset]);
            }
            break;
        }
        case 16: {
            if (is_bf16) {
                h->uni_vpmovzxwd(ymm, ptr[reg + offset]);
                h->uni_vpslld(ymm, ymm, 16);
            } else if (is_f16) {
                h->vcvtph2ps(ymm, ptr[reg + offset]);
            } else {
                if (is_signed)
                    h->uni_vpmovsxwd(ymm, ptr[reg + offset]);
                else
                    h->uni_vpmovzxwd(ymm, ptr[reg + offset]);
            }
            break;
        }
        case 8: {
            if (is_bf16) {
                h->uni_vpmovzxwd(xmm, ptr[reg + offset]);
                h->uni_vpslld(xmm, xmm, 16);
            } else if (is_f16) {
                h->vcvtph2ps(xmm, ptr[reg + offset]);
            } else {
                if (is_signed)
                    h->uni_vpmovsxwd(xmm, ptr[reg + offset]);
                else
                    h->uni_vpmovzxwd(xmm, ptr[reg + offset]);
            }
            break;
        }
        default: {
            if (is_zmm && load_size > threshold_for_mask_emu_load) {
                unsigned int mask = 1;
                mask = (mask << (load_size / 2)) - mask;
                h->mov(Reg32(aux_gpr_idxs[0]), mask);
                h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
                if (is_bf16) {
                    h->uni_vpmovzxwd(vmm | k_mask | T_z, ptr[reg + offset]);
                    h->uni_vpslld(vmm, vmm, 16);
                } else if (is_f16) {
                    h->vcvtph2ps(vmm | k_mask | T_z, ptr[reg + offset]);
                } else {
                    if (is_signed)
                        h->uni_vpmovsxwd(vmm | k_mask | T_z, ptr[reg + offset]);
                    else
                        h->uni_vpmovzxwd(vmm | k_mask | T_z, ptr[reg + offset]);
                }
            } else {
                // xmm or ymm version
                load_bytes(xmm, reg, offset, load_size);
                if (is_bf16) {
                    h->uni_vpmovzxwd(vmm, xmm);
                    h->uni_vpslld(vmm, vmm, 16);
                } else if (is_f16) {
                    h->vcvtph2ps(ymm, xmm);
                } else {
                    if (is_signed)
                        h->uni_vpmovsxwd(vmm, xmm);
                    else
                        h->uni_vpmovzxwd(vmm, xmm);
                }
            }
            break;
        }
    }
}

template <typename Vmm>
void jit_load_emitter::fill_with_default(const Vmm &vmm, std::string fill_value, const int &load_num) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    if (is_xmm || is_ymm) {
        uint8 imm = 1;
        imm = ~((imm << load_num) - imm);  // shift load_num bit
        h->uni_vblendps(vmm, vmm, table_val(fill_value), imm);
    } else if (is_zmm) {
        uint64_t tail_mask = 1;
        tail_mask = ~((tail_mask << load_num) - tail_mask);
        h->mov(Reg64(aux_gpr_idxs[0]), tail_mask);
        h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
        h->vblendmps(vmm | k_mask, vmm, table_val(fill_value));
    }
}

void jit_load_emitter::register_table_entries() {
    if (is_fill_) {
        push_arg_entry_of("zero", 0x00000000, true);
        push_arg_entry_of("int_one", 0x00000001, true);
        push_arg_entry_of("float_one", 0x3f800000, true);
        push_arg_entry_of("int32_min", 0xcf000000, true);
        push_arg_entry_of("float_min", 0xff7fffff, true);
        push_arg_entry_of("int32_max", 0x4effffff, true);
        push_arg_entry_of("float_max", 0x7f7fffff, true);
    }
}

/// STORE ///
jit_store_emitter::jit_store_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                     ov::element::Type src_prc, ov::element::Type dst_prc, int store_num, arithmetic_mode mode, ov::element::Type exec_prc,
                                     emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type), name_("unknown"), store_num_(store_num), src_prc_(src_prc), dst_prc_(dst_prc), mode_(mode) {
    prepare_table();
    v_len_elt_ = get_vec_length() / exec_prc.size();
    store_size_ = store_num * dst_prc.size();
    uni_vcvtneps2bf16_.reset(new jit_uni_vcvtneps2bf16(host, host_isa));
}

inline bool jit_store_emitter::is_saturation() const {
    return mode_ == arithmetic_mode::saturation;
}

// case for SSE and AVX2 when we should use AND to truncate values
inline bool jit_store_emitter::is_truncation_emulation() const {
    return !mayiuse(cpu::x64::avx512_core) && !is_saturation() &&
        src_prc_ != dst_prc_ && one_of(dst_prc_, ov::element::u16, ov::element::i16, ov::element::u8, ov::element::i8);
}

size_t jit_store_emitter::aux_gprs_count() const {
    // for temp reg for store(mask version or special number cases)
    const auto is_pure_store = (src_prc_ == dst_prc_) ||
                                (one_of(src_prc_, ov::element::f32, ov::element::i32) &&
                                 one_of(dst_prc_, ov::element::f32, ov::element::i32));
    const auto is_store_as_real16 = one_of(dst_prc_, ov::element::bf16, ov::element::f16);
    int count = get_aux_regs_as_temp(store_num_, static_cast<int>(dst_prc_.size()), is_pure_store, is_store_as_real16, threshold_for_mask_emu_store);

    // for table value in truncation arithmetic mode
    if (is_truncation_emulation())
        count++;

    return count;
}

size_t jit_store_emitter::aux_vecs_count() const {
    int count = 0;

    // to avoid src vmm pollution for data type conversion
    // and other vmm data pollution instructions
    if (src_prc_ != dst_prc_ || !one_of(store_size_, 64, 32, 16))
        count++;

    // for data swapping to avoid using Xmm(0) as I/O xmm for jit_uni_vcvtneps2bf16
    if ((host_isa_ == cpu::x64::sse41) && (src_prc_ == ov::element::f32 && dst_prc_ == ov::element::bf16))
        count++;

    // zero value, zeroed and passed from caller from performance standpoint(zeroed one time and not need preserve and restore status)
    if (mayiuse(cpu::x64::avx512_core) && one_of(dst_prc_, ov::element::u8, ov::element::u16))
        count++;

    return count;
}

size_t jit_store_emitter::get_inputs_num() const { return 1; }

void jit_store_emitter::emit_data() const {
    jit_emitter::emit_data();
    if (uni_vcvtneps2bf16_)
        uni_vcvtneps2bf16_->emit_data();
}

void jit_store_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    // offset in store emitter is the offset of dst gpr register, should be parsed from out_idxs.
    const int offset = out_idxs.size() == 2 ? out_idxs[1] : 0;
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(static_cast<int>(in_idxs[0]), Reg64(out_idxs[0]), offset);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(static_cast<int>(in_idxs[0]), Reg64(out_idxs[0]), offset);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        emit_isa<cpu::x64::avx512_core>(static_cast<int>(in_idxs[0]), Reg64(out_idxs[0]), offset);
    } else {
        OV_CPU_JIT_EMITTER_THROW("is performed on unsupported isa(at least x64::sse41).");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_store_emitter::emit_isa(const int in_vec_idx, const Xbyak::Reg64 &reg_dst, const int offset) const {
    bool matched_prc = (src_prc_ == dst_prc_) || (src_prc_ == ov::element::f32) || (src_prc_ == ov::element::i32);
    if (!matched_prc) {
        OV_CPU_JIT_EMITTER_THROW("only support input precision of FP32 or I32 or the same precision as output.");
    }
    if ((src_prc_ == ov::element::f32) || (src_prc_ == ov::element::i32)) {
        if ((isa == cpu::x64::sse41 && store_num_ > 4) || (isa == cpu::x64::avx2 && store_num_ > 8) ||
            (isa == cpu::x64::avx512_core && store_num_ > 16) || store_num_ < 0) {
            OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store.");
        }
    }
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;

    data_idx = in_vec_idx;
    data_reg_updated = false;
    if (!aux_vec_idxs.empty())
        aux_src_idx = aux_vec_idxs.back(); // to avoid src pollution
    if (src_prc_ != dst_prc_) {
        switch (src_prc_) {
            case ov::element::f32:
                if (!dst_prc_.is_real()) {
                    if (is_saturation()) {
                        h->uni_vcvtps2dq(Vmm(aux_src_idx), Vmm(data_idx));
                    } else {
                        h->uni_vcvttps2dq(Vmm(aux_src_idx), Vmm(data_idx));
                    }
                    data_idx = aux_src_idx;
                    data_reg_updated = true;
                }
                break;
            case ov::element::i32:
                if (dst_prc_.is_real()) {
                    h->uni_vcvtdq2ps(Vmm(aux_src_idx), Vmm(data_idx));
                    data_idx = aux_src_idx;
                    data_reg_updated = true;
                }
                break;
            default:
                break;
        }
    }

    if (src_prc_ == dst_prc_) {
        store_bytes<Vmm>(reg_dst, offset, store_size_);
    } else {
        switch (dst_prc_) {
            case ov::element::f32:
            case ov::element::i32:
                store_bytes<Vmm>(reg_dst, offset, store_size_);
                break;
            case ov::element::i8:
                store_dword_to_byte_extension<Vmm>(reg_dst, offset, true, store_num_);
                break;
            case ov::element::u8:
                store_dword_to_byte_extension<Vmm>(reg_dst, offset, false, store_num_);
                break;
            case ov::element::i16:
            case ov::element::u16:
            case ov::element::bf16:
            case ov::element::f16:
                store_dword_to_word_extension<Vmm>(reg_dst, offset, dst_prc_, store_num_);
                break;
            default:
                OV_CPU_JIT_EMITTER_THROW("has unsupported dst precision to store.");
        }
    }
}
/**
* store_bytes is the utility function to facilitate storing of
* store_size (0 <= store_size <= 64) many contiguous bytes from the Xmm/Ymm/Zmm
* register into the memory referenced by ptr[reg + offset] address.
*
* Additionally, when store_size > 16, the input Ymm register will not be
* preserved due to the usage of vextracti128 instruction.
*
* Functionally, invocation of store_bytes is equivalent
* to the following loop:
*
* for (int idx = 0; idx < store_size; ++idx)
*     vpextrb(ptr[reg + offset + idx], vmm, idx);
*
*/
template <typename Vmm>
void jit_store_emitter::store_bytes(const Xbyak::Reg64 &reg, int offset, int store_size) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure data fits completely inside the Xmm/Ymm/Zmm register
    if (store_size < 0 || store_size > 64)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store in store_bytes.");
    if (is_ymm && store_size > 32)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store to ymm in store_bytes.");
    if (is_xmm && store_size > 16)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store to xmm in store_bytes.");

    auto xmm = Xbyak::Xmm(data_idx);
    auto ymm = Xbyak::Ymm(data_idx);
    auto zmm = Xbyak::Zmm(data_idx);
    auto vmm = Vmm(data_idx);

    const auto addr = [&](int bytes_offset) {
        return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
    };

    auto store_byte_base = [&]() {
        int start_bytes = 0;
        int bytes_to_store = store_size;

        if (store_size > 32) {
            h->uni_vmovdqu(addr(0), ymm); // store lower bits from zmm
            start_bytes += 32;
            bytes_to_store -= 32;
            // load upper bits from zmm into ymm
            STORE_KEEP_SOURCE(vextractf64x4, ymm, Ymm(aux_src_idx), zmm, 1);
        }

        if (bytes_to_store > 16) {
            h->uni_vmovdqu(addr(start_bytes), xmm); // store lower bits from ymm
            start_bytes += 16;
            bytes_to_store -= 16;
            // load upper bits from ymm into xmm
            STORE_KEEP_SOURCE(vextractf128, xmm, Xmm(aux_src_idx), ymm, 1);
        }

        if (bytes_to_store >= 8 && bytes_to_store < 16)
            h->uni_vmovq(addr(start_bytes), xmm);
        else if (bytes_to_store == 16)
            h->uni_vmovdqu(addr(start_bytes), xmm);

        // 64/32/16/8 with one go
        // tail 7 bytes for lower or upper xmm
        auto store_one_byte = [&](int bytes_offset, int gpr_idx) {
            bool ext8bit = false;
            if (one_of(gpr_idx, Operand::RSP, Operand::RBP, Operand::RSI, Operand::RDI))
                ext8bit = true;
            h->mov(addr(start_bytes + bytes_offset), Reg8(gpr_idx, ext8bit));
        };
        switch (bytes_to_store) {
            case 0: break;
            case 1:
                h->uni_vmovq(Reg64(aux_gpr_idxs[0]), xmm);
                store_one_byte(0, aux_gpr_idxs[0]);
                break;
            case 2:
                h->uni_vmovq(Reg64(aux_gpr_idxs[0]), xmm);
                h->mov(addr(start_bytes), Reg16(aux_gpr_idxs[0]));
                break;
            case 3:
                h->uni_vmovq(Reg64(aux_gpr_idxs[0]), xmm);
                h->mov(addr(start_bytes), Reg16(aux_gpr_idxs[0]));
                h->shr(Reg64(aux_gpr_idxs[0]), 16);
                store_one_byte(2, aux_gpr_idxs[0]);
                break;
            case 4: h->uni_vmovss(addr(start_bytes), xmm); break;
            case 5:
                h->uni_vmovss(addr(start_bytes), xmm);
                h->uni_vpextrb(addr(start_bytes + 4), xmm, 4);
                break;
            case 6:
                h->uni_vmovss(addr(start_bytes), xmm);
                h->uni_vpextrw(addr(start_bytes + 4), xmm, 2);
                break;
            case 7:
                h->uni_vmovss(addr(start_bytes), xmm);
                h->uni_vpextrw(addr(start_bytes + 4), xmm, 2);
                h->uni_vpextrb(addr(start_bytes + 6), xmm, 6);
                break;
            case 8: break;
            case 9: h->uni_vpextrb(addr(start_bytes + 8), xmm, 8); break;
            case 10: h->uni_vpextrw(addr(start_bytes + 8), xmm, 4); break;
            case 11:
                h->uni_vpextrw(addr(start_bytes + 8), xmm, 4);
                h->uni_vpextrb(addr(start_bytes + 10), xmm, 10);
                break;
            case 12: h->uni_vpextrd(addr(start_bytes + 8), xmm, 2); break;
            case 13:
                h->uni_vpextrd(addr(start_bytes + 8), xmm, 2);
                h->uni_vpextrb(addr(start_bytes + 12), xmm, 12);
                break;
            case 14:
                h->uni_vpextrd(addr(start_bytes + 8), xmm, 2);
                h->uni_vpextrw(addr(start_bytes + 12), xmm, 6);
                break;
            case 15:
                h->uni_vpextrd(addr(start_bytes + 8), xmm, 2);
                h->uni_vpextrw(addr(start_bytes + 12), xmm, 6);
                h->uni_vpextrb(addr(start_bytes + 14), xmm, 14);
                break;
            case 16: break;
            default:
                OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store in store_bytes.");
        }
    };

    switch (store_size) {
        case 64:
            h->uni_vmovdqu(addr(0), zmm);
            break;
        case 32:
            h->uni_vmovdqu(addr(0), ymm);
            break;
        case 16:
            h->uni_vmovdqu(addr(0), xmm);
            break;
        default:
            if (mayiuse(cpu::x64::avx512_core) && store_size > threshold_for_mask_emu_store) {
                uint64_t mask = 1;
                mask = (mask << store_size) - mask;
                h->mov(Reg64(aux_gpr_idxs[0]), mask);
                h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
                h->vmovdqu8(addr(0), zmm | k_mask);
            } else {
                store_byte_base();
            }
            break;
    }
}

/**
* store_dword_to_byte_extension is the utility function to
* 1. convert store_num (0 <= store_num <= 16) dwords in the Xmm/Ymm/Zmm to store_num bytes, singed or unsinged, truncated or saturated.
* 2. store the packed byte into the memory referenced by ptr[reg + offset] address.
*/
template <typename Vmm>
void jit_store_emitter::store_dword_to_byte_extension(const Xbyak::Reg64 &reg, int offset, bool is_signed, int store_num) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure data fits completely inside the Xmm/Ymm/Zmm register
    // At most 8 dwords can fit inside the Ymm register
    // At most 4 dwords can fit inside the Xmm register
    if (store_num < 0 || store_num > 16)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store in store_dword_to_byte_extension.");
    if (is_ymm && store_num > 8)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store to ymm in store_dword_to_byte_extension.");
    if (is_xmm && store_num > 4)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store to xmm in store_dword_to_byte_extension.");

    auto vmm = Vmm(data_idx);
    auto zmm = Xbyak::Zmm(data_idx);
    auto ymm = Xbyak::Ymm(data_idx);
    auto xmm = Xbyak::Xmm(data_idx);

    const auto addr = [&](int bytes_offset) {
        return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
    };

    auto store_dword_to_byte_base = [&]() {
        if (is_zmm) {
            if (is_saturation()) {
                if (is_signed) {
                    STORE_KEEP_SOURCE(vpmovsdb, xmm, Xmm(aux_src_idx), vmm);
                } else {
                    Vmm zero(aux_vec_idxs[0]);
                    h->uni_vpxor(zero, zero, zero);
                    STORE_KEEP_SOURCE(uni_vpmaxsd, vmm, Vmm(aux_src_idx), vmm, zero);
                    h->vpmovusdb(xmm, vmm);
                }
            } else {
                STORE_KEEP_SOURCE(vpmovdb, xmm, Xmm(aux_src_idx), vmm);
            }
        } else {
            if (is_saturation()) {
                // db only available on avx512, need dw+wb to emulate
                if (is_signed) {
                    STORE_KEEP_SOURCE(uni_vpackssdw, vmm, Vmm(aux_src_idx), vmm, vmm);
                } else {
                    STORE_KEEP_SOURCE(uni_vpackusdw, vmm, Vmm(aux_src_idx), vmm, vmm);
                }
                // gather 2(cross lane) 64 bits into lower vmm to store when store_num > 4.
                // [y_3 y_2 y_1 y_0] |--> [y_0 y_0 y_2 y_0]
                if (is_ymm && (store_num > 4)) {
                    // 0x08:00001000
                    STORE_KEEP_SOURCE(vpermq, ymm, Ymm(aux_src_idx), ymm, 0x08);
                }

                if (is_signed) {
                    STORE_KEEP_SOURCE(uni_vpacksswb, vmm, Vmm(aux_src_idx), vmm, vmm);
                } else {
                    STORE_KEEP_SOURCE(uni_vpackuswb, vmm, Vmm(aux_src_idx), vmm, vmm);
                }
            } else {
                // to avoid saturation
                STORE_KEEP_SOURCE(vpand, vmm, Vmm(aux_src_idx), vmm, table_val("mask_truncation_byte"));
                STORE_KEEP_SOURCE(uni_vpackssdw, vmm, Vmm(aux_src_idx), vmm, vmm);
                if (is_ymm) {
                    STORE_KEEP_SOURCE(vpermq, ymm, Ymm(aux_src_idx), ymm, 0x08);
                }
                STORE_KEEP_SOURCE(uni_vpackuswb, vmm, Vmm(aux_src_idx), vmm, vmm);
            }
        }

        store_bytes<Vmm>(reg, offset, store_num);
    };

    switch (store_num) {
    case 16:
        // must support avx512F
        if (is_saturation()) {
            if (is_signed) {
                h->vpmovsdb(addr(0), vmm);
            } else {
                Vmm zero(aux_vec_idxs[0]);
                h->uni_vpxor(zero, zero, zero);
                STORE_KEEP_SOURCE(uni_vpmaxsd, vmm, Vmm(aux_src_idx), vmm, zero);
                h->vpmovusdb(addr(0), vmm);
            }
        } else {
            h->vpmovdb(addr(0), vmm);
        }
        break;
    case 8:
        if (mayiuse(cpu::x64::avx512_core)) {
            if (is_saturation()) {  // ymm block on avx512F + VL
                if (is_signed) {
                    h->vpmovsdb(addr(0), ymm);
                } else {
                    Vmm zero(aux_vec_idxs[0]);
                    h->uni_vpxor(zero, zero, zero);
                    STORE_KEEP_SOURCE(uni_vpmaxsd, ymm, Ymm(aux_src_idx), ymm, zero);
                    h->vpmovusdb(addr(0), ymm);
                }
            } else {
                h->vpmovdb(addr(0), ymm);
            }
        } else {
            store_dword_to_byte_base();
        }
        break;
    case 4:
        if (mayiuse(cpu::x64::avx512_core)) {
            if (is_saturation()) { // xmm block on avx512F + VL
                if (is_signed) {
                    h->vpmovsdb(addr(0), xmm);
                } else {
                    Vmm zero(aux_vec_idxs[0]);
                    h->uni_vpxor(zero, zero, zero);
                    STORE_KEEP_SOURCE(uni_vpmaxsd, xmm, Xmm(aux_src_idx), xmm, zero);
                    h->vpmovusdb(addr(0), xmm);
                }
            } else {
                h->vpmovdb(addr(0), xmm);
            }
        } else {
            store_dword_to_byte_base();
        }
        break;
    default:
        if (is_zmm && store_num > threshold_for_mask_emu_store) {  // avx512F
            unsigned int mask = 1;
            mask = (mask << store_num) - mask;
            h->mov(Reg32(aux_gpr_idxs[0]), mask);
            h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
            if (is_saturation()) {
                if (is_signed) {
                    h->vpmovsdb(addr(0), vmm | k_mask);
                } else {
                    Vmm zero(aux_vec_idxs[0]);
                    h->uni_vpxor(zero, zero, zero);
                    STORE_KEEP_SOURCE(uni_vpmaxsd, vmm, Vmm(aux_src_idx), vmm, zero);
                    h->vpmovusdb(addr(0), vmm | k_mask);
                }
            } else {
                h->vpmovdb(addr(0), vmm | k_mask);
            }
        } else {
            store_dword_to_byte_base();
        }
        break;
    }
}

/**
* store_dword_to_word_extension is the utility function to
* 1. convert store_num (0 <= store_num <= 16) dwords in the Xmm/Ymm/Zmm to store_num words with singed or unsinged saturation.
* 2. store the packed words into the memory referenced by ptr[reg + offset] address.
*/
template <typename Vmm>
void jit_store_emitter::store_dword_to_word_extension(const Xbyak::Reg64 &reg,
    int offset, ov::element::Type precision, int store_num) const {
    const bool is_bf16 = (precision == ov::element::bf16);
    const bool is_f16 = (precision == ov::element::f16);
    const bool is_signed = precision.is_signed();

    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure data fits completely inside the Xmm/Ymm/Zmm register
    // At most 4 dwords can fit inside the Xmm register
    // At most 8 dwords can fit inside the Ymm register
    if (store_num < 0 || store_num > 16)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store in store_dword_to_word_extension.");
    if (is_ymm && store_num > 8)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store to ymm in store_dword_to_word_extension.");
    if (is_xmm && store_num > 4)
        OV_CPU_JIT_EMITTER_THROW("has unexpected number of values to store to xmm in store_dword_to_word_extension.");

    auto xmm = Xbyak::Xmm(data_idx);
    auto ymm = Xbyak::Ymm(data_idx);
    auto zmm = Xbyak::Zmm(data_idx);
    auto vmm = Vmm(data_idx);

    auto store_dword_to_word_base = [&]() {
        if (is_zmm) {
            if (is_saturation()) {
                if (is_signed) {
                    // singed int32 saturate to signed int16.
                    STORE_KEEP_SOURCE(vpmovsdw, ymm, Ymm(aux_src_idx), vmm);
                } else {
                    // unsinged int32 saturate to unsigned int16.
                    Vmm zero(aux_vec_idxs[0]);
                    h->uni_vpxor(zero, zero, zero);
                    STORE_KEEP_SOURCE(uni_vpmaxsd, vmm, Vmm(aux_src_idx), vmm, zero);
                    STORE_KEEP_SOURCE(vpmovusdw, ymm, Ymm(aux_src_idx), vmm);
                }
            } else {
                // by literally copy low 16 bit
                STORE_KEEP_SOURCE(vpmovdw, ymm, Ymm(aux_src_idx), vmm);
            }
        } else {
            // direct mov_dw available only on avx512
            if (is_saturation()) {  // emulate with pack_dw + permute + pure store for saturation mode
                if (is_signed) {
                    STORE_KEEP_SOURCE(uni_vpackssdw, vmm, Vmm(aux_src_idx), vmm, vmm);
                } else {
                    STORE_KEEP_SOURCE(uni_vpackusdw, vmm, Vmm(aux_src_idx), vmm, vmm);
                }
                // gather 2/4(cross lane) 64 bits into lower vmm to store when store_num > 4
                // [y_3 y_2 y_1 y_0] |--> [y_0 y_0 y_2 y_0]
                // [  128  |  128  ] |--> [ 128   |  128  ]
                if (is_ymm && (store_num > 4)) {
                    // 0x08:00001000
                    STORE_KEEP_SOURCE(vpermq, ymm, Ymm(aux_src_idx), ymm, 0x08);
                }
            } else {  // emulate with AND + pure store for truncation mode
                STORE_KEEP_SOURCE(vpand, vmm, Vmm(aux_src_idx), vmm, table_val("mask_truncation_word"));
                STORE_KEEP_SOURCE(uni_vpackusdw, vmm, Vmm(aux_src_idx), vmm, vmm);
            }
        }

        store_bytes<Vmm>(reg, offset, store_num * 2);
    };

    if (is_bf16) {
        if (mayiuse(cpu::x64::avx512_core)) {
            // to avoid src vmm pollution, this check means no precision convert happens, so data_idx is still original_data_idx.
            if (src_prc_ == ov::element::f32) {
                ymm = Ymm(aux_vec_idxs[0]);
            }
            uni_vcvtneps2bf16_->emit_code({static_cast<size_t>(zmm.getIdx())}, {static_cast<size_t>(ymm.getIdx())});
            if (store_num == 16) {
                h->vmovdqu16(ptr[reg + offset], ymm);
            } else {
                data_idx = static_cast<int>(ymm.getIdx());
                store_bytes<Vmm>(reg, offset, store_num * 2);
            }
        } else {
            // to avoid src vmm pollution
            if (src_prc_ == ov::element::f32) {
                xmm = Xmm(aux_vec_idxs[0]);
            }
            // For sse41 mask register has to be Xmm(0) so we cannot use Xmm(0) as I/O vmm in uni_vcvtneps2bf16_
            if (host_isa_ == cpu::x64::sse41 && src_prc_ == ov::element::f32) {
                auto xmm_aux1 = Xmm(aux_vec_idxs[1]);
                h->uni_vmovups(xmm_aux1, vmm);
                uni_vcvtneps2bf16_->emit_code({static_cast<size_t>(vmm.getIdx())}, {static_cast<size_t>(vmm.getIdx())},
                                              {static_cast<size_t>(xmm.getIdx())});
                h->uni_vmovups(xmm, vmm);
                h->uni_vmovups(vmm, xmm_aux1);  // return original data to src vmm
            } else {
                uni_vcvtneps2bf16_->emit_code({static_cast<size_t>(vmm.getIdx())}, {static_cast<size_t>(xmm.getIdx())});
            }

            data_idx = static_cast<int>(xmm.getIdx());
            store_bytes<Vmm>(reg, offset, store_num * 2);
        }
    } else if (is_f16) {
        if (mayiuse(cpu::x64::avx512_core)) {
            // to avoid src vmm pollution
            if (src_prc_ == ov::element::f32) {
                // since avx512, zmm(fp32) => ymm(fp16)
                ymm = Ymm(aux_vec_idxs[0]);
            }  // in I32 case, zmm&ymm is already in aux reg

            h->vcvtps2ph(ymm, zmm, 0x4);
            if (store_num == 16) {
                h->vmovdqu16(ptr[reg + offset], ymm);
            } else {
                data_idx = static_cast<int>(ymm.getIdx());
                store_bytes<Vmm>(reg, offset, store_num * 2);
            }
        } else if (mayiuse(cpu::x64::avx2)) {
            // to avoid src vmm pollution
            if (src_prc_ == ov::element::f32) {
                xmm = Xmm(aux_vec_idxs[0]);
            }
            h->vcvtps2ph(xmm, ymm, 0x4);
            if (store_num == 8) {
                h->uni_vmovdqu(ptr[reg + offset], xmm);
            } else {
                data_idx = static_cast<int>(xmm.getIdx());
                store_bytes<Vmm>(reg, offset, store_num * 2);
            }
        } else {
            OV_CPU_JIT_EMITTER_THROW("only support fp16 on platform with avx512_core or avx2.");
        }
    } else {
        switch (store_num) {
        case 16:
            if (is_saturation()) {
                if (is_signed) {
                    h->vpmovsdw(ptr[reg + offset], vmm);  // singed int32 saturate to signed int16.
                } else {
                    Vmm zero(aux_vec_idxs[0]);
                    h->uni_vpxor(zero, zero, zero);
                    STORE_KEEP_SOURCE(uni_vpmaxsd, vmm, Vmm(aux_src_idx), vmm, zero);
                    h->vpmovusdw(ptr[reg + offset], vmm); // unsinged int32 saturate to unsigned int16.
                }
            } else {
                h->vpmovdw(ptr[reg + offset], vmm);
            }
            break;
        case 8:
            if (mayiuse(cpu::x64::avx512_core)) {
                if (is_saturation()) {
                    if (is_signed) {
                        h->vpmovsdw(ptr[reg + offset], ymm);
                    } else {
                        Vmm zero(aux_vec_idxs[0]);
                        h->uni_vpxor(zero, zero, zero);
                        STORE_KEEP_SOURCE(uni_vpmaxsd, ymm, Ymm(aux_src_idx), ymm, zero);
                        h->vpmovusdw(ptr[reg + offset], ymm);
                    }
                } else {
                    h->vpmovdw(ptr[reg + offset], ymm);
                }
            } else {
                store_dword_to_word_base();
            }
            break;
        case 4:
            if (mayiuse(cpu::x64::avx512_core)) {
                if (is_saturation()) {
                    if (is_signed) {
                        h->vpmovsdw(ptr[reg + offset], xmm);
                    } else {
                        Vmm zero(aux_vec_idxs[0]);
                        h->uni_vpxor(zero, zero, zero);
                        STORE_KEEP_SOURCE(uni_vpmaxsd, xmm, Xmm(aux_src_idx), xmm, zero);
                        h->vpmovusdw(ptr[reg + offset], xmm);
                    }
                } else {
                    h->vpmovdw(ptr[reg + offset], xmm);
                }
            } else {
               store_dword_to_word_base();
            }
            break;
        default:
            if (is_zmm && ((store_num * 2) > threshold_for_mask_emu_store)) {
                unsigned int mask = 1;
                mask = (mask << store_num) - mask;
                h->mov(Reg32(aux_gpr_idxs[0]), mask);
                h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
                if (is_saturation()) {
                    if (is_signed) {
                        h->vpmovsdw(ptr[reg + offset], vmm | k_mask);
                    } else {
                        Vmm zero(aux_vec_idxs[0]);
                        h->uni_vpxor(zero, zero, zero);
                        STORE_KEEP_SOURCE(uni_vpmaxsd, vmm, Vmm(aux_src_idx), vmm, zero);
                        h->vpmovusdw(ptr[reg + offset], vmm | k_mask);
                    }
                } else {
                    h->vpmovdw(ptr[reg + offset], vmm | k_mask);
                }
            } else {
                store_dword_to_word_base();
            }
            break;
        }
    }
}

void jit_store_emitter::register_table_entries() {
    if (is_truncation_emulation()) {
        push_arg_entry_of("mask_truncation_byte", 0x000000ff, true);
        push_arg_entry_of("mask_truncation_word", 0x0000ffff, true);
    }
}

}   // namespace intel_cpu
}   // namespace ov
