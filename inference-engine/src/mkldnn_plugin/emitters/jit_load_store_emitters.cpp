// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"
#include "jit_load_store_emitters.hpp"
#include <cpu/x64/jit_generator.hpp>
#include "utils/bfloat16.hpp"

using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;
using namespace Xbyak::util;

namespace MKLDNNPlugin {

/// LOAD ///
jit_load_emitter::jit_load_emitter(jit_generator *host, cpu_isa_t host_isa, const MKLDNNNode* node,
    Precision exec_prc, emitter_in_out_map in_out_type)
: jit_emitter(host, host_isa, node, exec_prc, in_out_type), name(node ? node->getName() : "unknown") {
    prepare_table();
    v_len_elt = get_vec_length() / exec_prc.size();
}

size_t jit_load_emitter::get_inputs_num() const { return 1; }

// 0 for temp reg for mask load, 1 for table address
size_t jit_load_emitter::aux_gprs_count() const {
    return 2;
}

void jit_load_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const {
    const auto* load_emitter_context = dynamic_cast<const MKLDNNPlugin::load_emitter_context*>(emit_context);
    if (load_emitter_context == nullptr) {
        IE_THROW() << "Load emitter in " << name << " does not get load emmiter context.";
    }

    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(Reg64(in_idxs[0]), load_emitter_context->offset_byte_, load_emitter_context->src_prc_, static_cast<int>(out_idxs[0]),
            load_emitter_context->dst_prc_, load_emitter_context->load_num_, load_emitter_context->is_fill_, load_emitter_context->fill_value_);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(Reg64(in_idxs[0]), load_emitter_context->offset_byte_, load_emitter_context->src_prc_, static_cast<int>(out_idxs[0]),
            load_emitter_context->dst_prc_, load_emitter_context->load_num_, load_emitter_context->is_fill_, load_emitter_context->fill_value_);
    } else if (host_isa_ == cpu::x64::avx512_common) {
        emit_isa<cpu::x64::avx512_common>(Reg64(in_idxs[0]), load_emitter_context->offset_byte_, load_emitter_context->src_prc_, static_cast<int>(out_idxs[0]),
            load_emitter_context->dst_prc_, load_emitter_context->load_num_, load_emitter_context->is_fill_, load_emitter_context->fill_value_);
    } else {
        IE_THROW() << "Load emitter in " << name << " is performed on unsupported isa(at least x64::sse41).";
    }
}

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
void jit_load_emitter::emit_isa(const Xbyak::Reg64 &reg_src, int offset_byte, InferenceEngine::Precision src_prc,
    const int out_vec_idx, InferenceEngine::Precision dst_prc, int load_num, bool is_fill, std::string fill_value) const {
    bool matched_prc = (dst_prc == src_prc) || (dst_prc == Precision::FP32) || (dst_prc == Precision::I32);
    if (!matched_prc) {
        IE_THROW() << "Load emitter in " << name << " only support output precision of FP32 or I32 or the same precision as input.";
    }
    if (load_num > (get_vec_length() / dst_prc.size())) {
        IE_THROW() << "Load emitter in " << name << " have unexpected number of elements to load.";
    }

    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;

    // pure load
    if (src_prc == dst_prc) {
        load_bytes<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, load_num * src_prc.size(), is_fill, fill_value);
    } else {
    // "pure load" + convert. dst_prc must be FP32 or I32.
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                load_bytes<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, load_num * src_prc.size(), is_fill, fill_value);
                break;
            case Precision::I8:
                load_bytes_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, true, load_num * src_prc.size(), is_fill, fill_value);
                break;
            case Precision::U8:
                load_bytes_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, false, load_num * src_prc.size(), is_fill, fill_value);
                break;
            case Precision::I16:
                load_words_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, false, true, load_num * src_prc.size(), is_fill, fill_value);
                break;
            case Precision::U16:
                load_words_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, false, false, load_num * src_prc.size(), is_fill, fill_value);
                break;
            case Precision::BF16:
                load_words_to_dword_extension<Vmm>(Vmm(out_vec_idx), reg_src, offset_byte, true, false, load_num * src_prc.size(), is_fill, fill_value);
                break;
            default:
                IE_THROW() << "Load emitter in " << name << " has unsupported src precision to load.";
        }
    }

    // post convert between I32 and FP32
    if (src_prc != dst_prc) {
        switch (dst_prc) {
            case Precision::FP32:
                if ((src_prc != Precision::FP32) && (src_prc != Precision::BF16))
                    h->uni_vcvtdq2ps(Vmm(out_vec_idx), Vmm(out_vec_idx));
                break;
            case Precision::I32:
                if ((src_prc == Precision::FP32) || (src_prc == Precision::BF16)) {
                    h->uni_vcvtps2dq(Vmm(out_vec_idx), Vmm(out_vec_idx));
                }
                break;
            default:
                break;
        }
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
void jit_load_emitter::load_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, int load_size,
    bool is_fill, std::string fill_value) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure data fits completely inside the Xmm/Ymm/Zmm register
    if (load_size < 0 || load_size > 64)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load in load_byte.";
    // check if proper number bytes fit inside the Xmm/Ymm register
    if (is_ymm && load_size > 32)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load to ymm in load_byte.";
    if (is_xmm && load_size > 16)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load to xmm in load_byte.";

    auto xmm = Xbyak::Xmm(vmm.getIdx());
    auto ymm = Xbyak::Ymm(vmm.getIdx());
    auto zmm = Xbyak::Zmm(vmm.getIdx());

    // addr(i) denotes the memory pointed by ptr[reg + offset + (i bytes)]
    const auto addr = [&](int bytes_offset) {
        return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
    };

    if (is_zmm && (load_size != 64) && mayiuse(cpu::x64::avx512_core)) {
        uint64_t mask = 1;
        mask = (mask << load_size) - mask;
        h->mov(Reg64(aux_gpr_idxs[0]), mask);
        h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
        h->vmovdqu8(zmm | k_mask | T_z, addr(0));
    } else {
        if (load_size == 64) {
            h->uni_vmovdqu(zmm, addr(0));
        } else if (load_size == 32) {
            h->uni_vmovdqu(ymm, addr(0));
        } else if (load_size == 16) {
            h->uni_vmovdqu(xmm, addr(0));
        } else {
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

            if (bytes_to_load >= 8 && bytes_to_load < 16)
                h->pinsrq(xmm, addr(start_bytes), 0);
            else if (bytes_to_load == 16)
                h->uni_vmovdqu(xmm, addr(start_bytes));

            switch (bytes_to_load) {
                case 0: break;
                case 1: h->uni_vpinsrb(xmm, xmm, addr(start_bytes), 0); break;
                case 2: h->uni_vpinsrw(xmm, xmm, addr(start_bytes), 0); break;
                case 3:
                    h->uni_vpinsrw(xmm, xmm, addr(start_bytes), 0);
                    h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 2), 2);
                    break;
                case 4: h->pinsrd(xmm, addr(start_bytes), 0); break;
                case 5:
                    h->pinsrd(xmm, addr(start_bytes), 0);
                    h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 4), 4);
                    break;
                case 6:
                    h->pinsrd(xmm, addr(start_bytes), 0);
                    h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 4), 2);
                    break;
                case 7:
                    h->pinsrd(xmm, addr(start_bytes), 0);
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
                case 12: h->pinsrd(xmm, addr(start_bytes + 8), 2); break;
                case 13:
                    h->pinsrd(xmm, addr(start_bytes + 8), 2);
                    h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 12), 12);
                    break;
                case 14:
                    h->pinsrd(xmm, addr(start_bytes + 8), 2);
                    h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 12), 6);
                    break;
                case 15:
                    h->pinsrd(xmm, addr(start_bytes + 8), 2);
                    h->uni_vpinsrw(xmm, xmm, addr(start_bytes + 12), 6);
                    h->uni_vpinsrb(xmm, xmm, addr(start_bytes + 14), 14);
                    break;
                case 16: break;
                default:
                    IE_THROW() << "Load emitter in " << name<< " has unexpected number of values to load in load_byte.";
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
        }
    }

    if (is_fill)
        fill_with_default(vmm, fill_value, load_size / 4);
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
void jit_load_emitter::load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg,
        int offset, bool is_signed, int load_size, bool is_fill, std::string fill_value) const {
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
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load in load_bytes_to_dword_extension.";
    if (is_ymm && load_size > 8)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load to ymm in load_bytes_to_dword_extension.";
    if (is_xmm && load_size > 4)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load to xmm in load_bytes_to_dword_extension.";

    // For load_size == 4/8/16, do load/extension in one go
    if (load_size == 16) {
        const auto zmm = Xbyak::Zmm(vmm.getIdx());
        if (is_signed)
            h->uni_vpmovsxbd(zmm, ptr[reg + offset]);
        else
            h->uni_vpmovzxbd(zmm, ptr[reg + offset]);
    } else if (load_size == 8) {
        // full size of ymm or ymm_block of zmm
        const auto ymm = Xbyak::Ymm(vmm.getIdx());
        if (is_signed)
            h->uni_vpmovsxbd(ymm, ptr[reg + offset]);
        else
            h->uni_vpmovzxbd(ymm, ptr[reg + offset]);
    } else if (load_size == 4) {
        // full size of xmm or xmm_block of ymm/zmm
        const auto xmm = Xbyak::Xmm(vmm.getIdx());
        if (is_signed)
            h->uni_vpmovsxbd(xmm, ptr[reg + offset]);
        else
            h->uni_vpmovzxbd(xmm, ptr[reg + offset]);
    } else {
        // tails process
        if (is_zmm) {
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
    }

    if (is_fill)
        fill_with_default(vmm, fill_value, load_size);
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
void jit_load_emitter::load_words_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg,
        int offset, bool is_bf16, bool is_signed, int load_size, bool is_fill, std::string fill_value) const {
    constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    MAYBE_UNUSED(is_xmm);
    MAYBE_UNUSED(is_ymm);
    MAYBE_UNUSED(is_zmm);

    // Ensure extended double words fit inside Zmm (32/2(num) * 32 <= 512)
    // For Ymm register, load capacity is halved (16/2(num) * 32 <= 128)
    // For Xmm register, load capacity is halved again (8/2(num) * 32 <= 128)
    if (load_size < 0 || load_size > 32)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load in load_words_to_dword_extension.";
    if (is_ymm && load_size > 16)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load to ymm in load_words_to_dword_extension.";
    if (is_xmm && load_size > 8)
        IE_THROW() << "Load emitter in " << name << " has unexpected number of values to load to xmm in load_words_to_dword_extension.";

    auto xmm = Xbyak::Xmm(vmm.getIdx());
    auto ymm = Xbyak::Ymm(vmm.getIdx());
    auto zmm = Xbyak::Zmm(vmm.getIdx());

    // For load_size == 32/16/8, do load/extension in one go
    // including xmm/ymm tail block for ymm/zmm, so explicite xmm/ymm/zmm
    if (load_size == 32) {
        if (is_bf16) {
            h->uni_vpmovzxwd(zmm, ptr[reg + offset]);
            h->uni_vpslld(zmm, zmm, 16);
        } else {
            if (is_signed)
                h->uni_vpmovsxwd(zmm, ptr[reg + offset]);
            else
                h->uni_vpmovzxwd(zmm, ptr[reg + offset]);
        }
    } else if (load_size == 16) {
        if (is_bf16) {
            h->uni_vpmovzxwd(ymm, ptr[reg + offset]);
            h->uni_vpslld(ymm, ymm, 16);
        } else {
            if (is_signed)
                h->uni_vpmovsxwd(ymm, ptr[reg + offset]);
            else
                h->uni_vpmovzxwd(ymm, ptr[reg + offset]);
        }
    } else if (load_size == 8) {
        if (is_bf16) {
            h->uni_vpmovzxwd(xmm, ptr[reg + offset]);
            h->uni_vpslld(xmm, xmm, 16);
        } else {
            if (is_signed)
                h->uni_vpmovsxwd(xmm, ptr[reg + offset]);
            else
                h->uni_vpmovzxwd(xmm, ptr[reg + offset]);
        }
    } else {
        if (is_zmm) {
            unsigned int mask = 1;
            mask = (mask << (load_size / 2)) - mask;
            h->mov(Reg32(aux_gpr_idxs[0]), mask);
            h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
            if (is_bf16) {
                h->uni_vpmovzxwd(vmm | k_mask | T_z, ptr[reg + offset]);
                h->uni_vpslld(vmm, vmm, 16);
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
            } else {
                if (is_signed)
                    h->uni_vpmovsxwd(vmm, xmm);
                else
                    h->uni_vpmovzxwd(vmm, xmm);
            }
        }
    }

    if (is_fill)
        fill_with_default(vmm, fill_value, load_size / 2);
}

template <typename Vmm>
    void jit_load_emitter::fill_with_default(const Vmm &vmm, std::string fill_value, const int &load_num) const {
        constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
        constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
        constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

        if (is_xmm || is_ymm) {
            uint8 imm = 1;
            imm = ~((imm << load_num) - imm);  // shift load_num bit
            if (is_xmm)
                h->blendps(vmm, table_val(fill_value), imm);
            else
                h->vblendps(vmm, vmm, table_val(fill_value), imm);
        } else if (is_zmm) {
            uint64_t tail_mask = 1;
            tail_mask = ~((tail_mask << load_num) - tail_mask);
            h->mov(Reg64(aux_gpr_idxs[0]), tail_mask);
            h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
            h->vblendmps(vmm | k_mask, vmm, table_val(fill_value));
        }
    }

void jit_load_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("int_one", 0x00000001, true);
    push_arg_entry_of("float_one", 0x3f800000, true);
    push_arg_entry_of("int32_min", 0xcf000000, true);
    push_arg_entry_of("float_min", 0xff7fffff, true);
    push_arg_entry_of("int32_max", 0x4effffff, true);
    push_arg_entry_of("float_max", 0x7f7fffff, true);
}

/// STORE ///
jit_store_emitter::jit_store_emitter(jit_generator *host, cpu_isa_t host_isa, const MKLDNNNode* node,
    Precision exec_prc, emitter_in_out_map in_out_type)
: jit_emitter(host, host_isa, node, exec_prc, in_out_type), name(node ? node->getName() : "unknown") {
    v_len_elt = get_vec_length() / exec_prc.size();
    if (!mayiuse(cpu::x64::avx512_core_bf16) && mayiuse(cpu::x64::avx512_core)) {
        emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(host, host_isa, nullptr));
    }
}

// 0 for temp reg for mask store
size_t jit_store_emitter::aux_gprs_count() const {
    return 1;
}

// zero value, zeroed and passed from caller from performance standpoint(zeroed one time and not need preserve and restore status)
size_t jit_store_emitter::aux_vecs_count() const {
    return 1;
}

size_t jit_store_emitter::get_inputs_num() const { return 1; }

void jit_store_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const {
    const auto* store_emitter_context = dynamic_cast<const MKLDNNPlugin::store_emitter_context*>(emit_context);
    if (store_emitter_context == nullptr) {
        IE_THROW() << "Store emitter in " << name << " does not get store emmiter context.";
    }
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(static_cast<int>(in_idxs[0]), store_emitter_context->src_prc_, Reg64(out_idxs[0]),
            store_emitter_context->offset_byte_, store_emitter_context->dst_prc_, store_emitter_context->store_num_);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(static_cast<int>(in_idxs[0]), store_emitter_context->src_prc_, Reg64(out_idxs[0]),
            store_emitter_context->offset_byte_, store_emitter_context->dst_prc_, store_emitter_context->store_num_);
    } else if (host_isa_ == cpu::x64::avx512_common) {
        emit_isa<cpu::x64::avx512_common>(static_cast<int>(in_idxs[0]), store_emitter_context->src_prc_, Reg64(out_idxs[0]),
            store_emitter_context->offset_byte_, store_emitter_context->dst_prc_, store_emitter_context->store_num_);
    } else {
        IE_THROW() << "Store emitter in " << name << " is performed on unsupported isa(at least x64::sse41).";
    }
}

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void jit_store_emitter::emit_isa(const int in_vec_idx, InferenceEngine::Precision src_prc,
        const Xbyak::Reg64 &reg_dst, int offset_byte, InferenceEngine::Precision dst_prc, int store_num) const {
        bool matched_prc = (src_prc == dst_prc) || (src_prc == Precision::FP32) || (src_prc == Precision::I32);
        if (!matched_prc) {
            IE_THROW() << "Store emitter in " << name << " only support input precision of FP32 or I32 or the same precision as output.";
        }
        if ((src_prc == Precision::FP32) || (src_prc == Precision::I32)) {
            if ((isa == cpu::x64::sse41 && store_num > 4) || (isa == cpu::x64::avx2 && store_num > 8) ||
                (isa == cpu::x64::avx512_common && store_num > 16) || store_num < 0) {
                IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store.";
            }
        }

        using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;

        if (src_prc != dst_prc) {
            switch (src_prc) {
                case Precision::FP32:
                    if ((dst_prc != Precision::FP32) && (dst_prc != Precision::BF16)) {
                        h->uni_vcvtps2dq(Vmm(in_vec_idx), Vmm(in_vec_idx));
                    }
                    break;
                case Precision::I32:
                    if ((dst_prc == Precision::FP32) || (dst_prc == Precision::BF16))
                        h->uni_vcvtdq2ps(Vmm(in_vec_idx), Vmm(in_vec_idx));
                    break;
                default:
                    break;
            }
        }

        if (src_prc == dst_prc) {
            store_bytes<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, store_num * dst_prc.size());
        } else {
            switch (dst_prc) {
                case Precision::FP32:
                case Precision::I32:
                    store_bytes<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, store_num * dst_prc.size());
                    break;
                case Precision::I8:
                    store_dword_to_byte_extension<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, true, store_num);
                    break;
                case Precision::U8:
                    store_dword_to_byte_extension<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, false, store_num);
                    break;
                case Precision::I16:
                    store_dword_to_word_extension<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, false, true, store_num);
                    break;
                case Precision::U16:
                    store_dword_to_word_extension<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, false, false, store_num);
                    break;
                case Precision::BF16:
                    store_dword_to_word_extension<Vmm>(Vmm(in_vec_idx), reg_dst, offset_byte, true, false, store_num);
                    break;
                default:
                    IE_THROW() << "Store emitter in " << name << " has unsupported dst precision to store.";
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
    void jit_store_emitter::store_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, int store_size) const {
        constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
        constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
        constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

        MAYBE_UNUSED(is_xmm);
        MAYBE_UNUSED(is_ymm);
        MAYBE_UNUSED(is_zmm);

        // Ensure data fits completely inside the Xmm/Ymm/Zmm register
        if (store_size < 0 || store_size > 64)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store in store_bytes.";
        if (is_ymm && store_size > 32)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store to ymm in store_bytes.";
        if (is_xmm && store_size > 16)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store to xmm in store_bytes.";

        auto xmm = Xbyak::Xmm(vmm.getIdx());
        auto ymm = Xbyak::Ymm(vmm.getIdx());
        auto zmm = Xbyak::Zmm(vmm.getIdx());

        const auto addr = [&](int bytes_offset) {
            return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
        };

        if (is_zmm && (store_size != 64) && mayiuse(cpu::x64::avx512_core)) {
            uint64_t mask = 1;
            mask = (mask << store_size) - mask;
            h->mov(Reg64(aux_gpr_idxs[0]), mask);
            h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
            h->vmovdqu8(addr(0), zmm | k_mask);
        } else {
            if (store_size == 64) {
                h->uni_vmovdqu(addr(0), zmm);
                return;
            } else if (store_size == 32) {
                h->uni_vmovdqu(addr(0), ymm);
                return;
            } else if (store_size == 16) {
                h->uni_vmovdqu(addr(0), xmm);
                return;
            } else {
                int start_bytes = 0;
                int bytes_to_store = store_size;

                if (store_size > 32) {
                    h->uni_vmovdqu(addr(0), ymm); // store lower bits from zmm
                    start_bytes += 32;
                    bytes_to_store -= 32;
                    h->vextractf64x4(ymm, zmm, 1); // load upper bits from zmm into ymm
                }

                if (bytes_to_store > 16) {
                    h->uni_vmovdqu(addr(start_bytes), xmm); // store lower bits from ymm
                    start_bytes += 16;
                    bytes_to_store -= 16;
                    h->vextractf128(xmm, ymm, 1); // load upper bits from ymm into xmm
                }

                if (bytes_to_store >= 8 && bytes_to_store < 16)
                    h->pextrq(addr(start_bytes), xmm, 0);
                else if (bytes_to_store == 16)
                    h->uni_vmovdqu(addr(start_bytes), xmm);

                // 64/32/16/8 with one go
                // tail 7 bytes for lower or upper xmm
                switch (bytes_to_store) {
                    case 0: break;
                    case 1: h->uni_vpextrb(addr(start_bytes), xmm, 0); break;
                    case 2: h->uni_vpextrw(addr(start_bytes), xmm, 0); break;
                    case 3:
                        h->uni_vpextrw(addr(start_bytes), xmm, 0);
                        h->uni_vpextrb(addr(start_bytes + 2), xmm, 2);
                        break;
                    case 4: h->pextrd(addr(start_bytes), xmm, 0); break;
                    case 5:
                        h->pextrd(addr(start_bytes), xmm, 0);
                        h->uni_vpextrb(addr(start_bytes + 4), xmm, 4);
                        break;
                    case 6:
                        h->pextrd(addr(start_bytes), xmm, 0);
                        h->uni_vpextrw(addr(start_bytes + 4), xmm, 2);
                        break;
                    case 7:
                        h->pextrd(addr(start_bytes), xmm, 0);
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
                    case 12: h->pextrd(addr(start_bytes + 8), xmm, 2); break;
                    case 13:
                        h->pextrd(addr(start_bytes + 8), xmm, 2);
                        h->uni_vpextrb(addr(start_bytes + 12), xmm, 12);
                        break;
                    case 14:
                        h->pextrd(addr(start_bytes + 8), xmm, 2);
                        h->uni_vpextrw(addr(start_bytes + 12), xmm, 6);
                        break;
                    case 15:
                        h->pextrd(addr(start_bytes + 8), xmm, 2);
                        h->uni_vpextrw(addr(start_bytes + 12), xmm, 6);
                        h->uni_vpextrb(addr(start_bytes + 14), xmm, 14);
                        break;
                    case 16: break;
                    default:
                        IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store in store_bytes.";
                }
            }
        }
    }

/**
* store_dword_to_byte_extension is the utility function to
* 1. convert store_num (0 <= store_num <= 16) dwords in the Xmm/Ymm/Zmm to store_num bytes with singed or unsinged saturation.
* 2. store the packed byte into the memory referenced by ptr[reg + offset] address.
*/
template <typename Vmm>
    void jit_store_emitter::store_dword_to_byte_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_signed, int store_num) const {
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
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store in store_dword_to_byte_extension.";
        if (is_ymm && store_num > 8)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store to ymm in store_dword_to_byte_extension.";
        if (is_xmm && store_num > 4)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store to xmm in store_dword_to_byte_extension.";

        auto ymm = Xbyak::Ymm(vmm.getIdx());

        const auto addr = [&](int bytes_offset) {
            return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
        };

        if (is_zmm) {
            if (store_num == 16) { // v_len_elt(16)
                if (is_signed) {
                    h->vpmovsdb(addr(0), vmm);
                } else {
                    h->vpmaxsd(vmm, vmm, Vmm(aux_vec_idxs[0]));
                    h->vpmovusdb(addr(0), vmm);
                }
            } else {
                unsigned int mask = 1;
                mask = (mask << store_num) - mask;
                h->mov(Reg32(aux_gpr_idxs[0]), mask);
                h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
                if (is_signed) {
                    h->vpmovsdb(addr(0), vmm | k_mask);
                } else {
                    h->vpmaxsd(vmm, vmm, Vmm(aux_vec_idxs[0]));
                    h->vpmovusdb(addr(0), vmm | k_mask);
                }
            }
        } else {
            // db only available on avx512, need dw+wb to emulate
            if (is_signed)
                h->uni_vpackssdw(vmm, vmm, vmm);
            else
                h->uni_vpackusdw(vmm, vmm, vmm);
            // gather 2(cross lane) 64 bits into lower vmm to store
            // [y_3 y_2 y_1 y_0] |--> [y_0 y_0 y_2 y_0]
            if (is_ymm) {
                h->vpermq(ymm, ymm, 0x08);  // 00001000
            }

            if (is_signed)
                h->uni_vpacksswb(vmm, vmm, vmm);
            else
                h->uni_vpackuswb(vmm, vmm, vmm);

            store_bytes(vmm, reg, offset, store_num);
        }
    }

/**
* store_dword_to_word_extension is the utility function to
* 1. convert store_num (0 <= store_num <= 16) dwords in the Xmm/Ymm/Zmm to store_num words with singed or unsinged saturation.
* 2. store the packed words into the memory referenced by ptr[reg + offset] address.
*/
template <typename Vmm>
    void jit_store_emitter::store_dword_to_word_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset,
        bool is_bf16, bool is_signed, int store_num) const {
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
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store in store_dword_to_word_extension.";
        if (is_ymm && store_num > 8)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store to ymm in store_dword_to_word_extension.";
        if (is_xmm && store_num > 4)
            IE_THROW() << "Store emitter in " << name << " has unexpected number of values to store to xmm in store_dword_to_word_extension.";

        auto ymm = Xbyak::Ymm(vmm.getIdx());
        auto zmm = Xbyak::Zmm(vmm.getIdx());

        if (is_bf16) {
            if (mayiuse(cpu::x64::avx512_core_bf16)) {
                h->vcvtneps2bf16(ymm, zmm);
            } else {
                emu_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm.getIdx())}, {static_cast<size_t>(ymm.getIdx())});
            }
            if (store_num == 16) {
                h->vmovdqu16(ptr[reg + offset], ymm);
            } else {
                store_bytes(ymm, reg, offset, store_num * 2);
            }
        } else {
            if (is_zmm) {
                if (store_num == 16) {  // v_len_elt
                    if (is_signed) {
                        h->vpmovsdw(ptr[reg + offset], vmm);  // singed int32 saturate to signed int16.
                    } else {
                        h->vmaxsd(vmm, Vmm(aux_vec_idxs[0]), vmm);        // if singed bit is 1, set value as 0.
                        h->vpmovusdw(ptr[reg + offset], vmm); // unsinged int32 saturate to unsigned int16.
                    }
                } else {
                    unsigned int mask = 1;
                    mask = (mask << store_num) - mask;
                    h->mov(Reg32(aux_gpr_idxs[0]), mask);
                    h->kmovw(k_mask, Reg32(aux_gpr_idxs[0]));
                    if (is_signed) {
                        h->vpmovsdw(ptr[reg + offset], vmm | k_mask);
                    } else {
                        h->vmaxsd(vmm, Vmm(aux_vec_idxs[0]), vmm);
                        h->vpmovusdw(ptr[reg + offset], vmm | k_mask);
                    }
                }
            } else {
                // direct mov_dw available only on avx512, emulate with pack_dw + permute + pure store
                if (is_signed)
                    h->uni_vpackssdw(vmm, vmm, vmm);
                else
                    h->uni_vpackusdw(vmm, vmm, vmm);
                // gather 2/4(cross lane) 64 bits into lower vmm to store
                // [y_3 y_2 y_1 y_0] |--> [y_0 y_0 y_2 y_0]
                // [  128  |  128  ] |--> [ 128   |  128  ]
                if (is_ymm) {
                    h->vpermq(ymm, ymm, 0x08);  // 00001000
                }

                store_bytes(vmm, reg, offset, store_num * 2);
            }
        }
    }

} // namespace MKLDNNPlugin
