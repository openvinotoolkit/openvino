// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_load_store_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator_t;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;

namespace {
template <typename TReg>
inline void cvt_f16_to_f32(jit_generator* h, const TReg& src, const TReg& dst) {
    h->fcvtl(dst.s4, src.h4);
}

template <typename TReg>
inline void cvt_f32_to_i32(jit_generator* h, const TReg& src, const TReg& dst, bool is_saturated) {
    if (is_saturated) {
        h->frintn(dst.s, src.s);
        h->fcvtzs(dst.s, dst.s);
    } else {
        h->fcvtzs(dst.s, src.s);
    }
}

template <typename TReg>
inline void cvt_i32_to_f32(jit_generator* h, const TReg& src, const TReg& dst) {
    h->scvtf(dst.s, src.s);
}

template <typename TReg>
inline void cvt_i32_to_i16(jit_generator* h, const TReg& src, const TReg& dst, bool is_saturated) {
    if (is_saturated) {
        h->sqxtn(dst.h4, src.s4);
    } else {
        h->xtn(dst.h4, src.s4);
    }
}

template <typename TReg>
inline void cvt_i16_to_i32(jit_generator* h, const TReg& src, const TReg& dst) {
    h->sxtl(dst.s4, src.h4);
}

template <typename TReg>
inline void cvt_f32_to_f16(jit_generator* h, const TReg& src, const TReg& dst) {
    h->fcvtn(dst.h4, src.s4);
}

template <typename TReg>
inline void cvt_byte_to_i16(jit_generator* h, const TReg& src, const TReg& dst, bool is_signed) {
    if (is_signed) {
        h->sxtl(dst.h8, src.b8);
    } else {
        h->uxtl(dst.h8, src.b8);
    }
}

template <typename TReg>
inline void cvt_i16_to_byte(jit_generator* h, const TReg& src, const TReg& dst, bool is_signed, bool is_saturated) {
    if (is_saturated) {
        if (is_signed) {
            h->sqxtn(dst.b8, src.h8);
        } else {
            h->uqxtn(dst.b8, src.h8);
        }
    } else {
        h->xtn(dst.b8, src.h8);
    }
}

template <typename TReg>
void jit_convert_process(jit_generator* h,
                         const TReg& src,
                         const TReg& dst,
                         ov::element::Type input_type,
                         ov::element::Type output_type,
                         bool is_saturated) {
    if (input_type == output_type || (!is_saturated && any_of(input_type, ov::element::i8, ov::element::u8) &&
                                      any_of(output_type, ov::element::i8, ov::element::u8))) {
        if (src.getIdx() != dst.getIdx()) {
            h->mov(dst.b16, src.b16);
        }
        return;
    }

    switch (output_type) {
    case ov::element::f32:
        switch (input_type) {
        case ov::element::i32:
            cvt_i32_to_f32(h, src, dst);
            break;
        case ov::element::f16:
            cvt_f16_to_f32(h, src, dst);
            break;
        case ov::element::i8:
        case ov::element::u8:
            cvt_byte_to_i16(h, src, dst, input_type.is_signed());
            cvt_i16_to_i32(h, dst, dst);
            cvt_i32_to_f32(h, dst, dst);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
        }
        break;
    case ov::element::i32:
        switch (input_type) {
        case ov::element::f32:
            cvt_f32_to_i32(h, src, dst, is_saturated);
            break;
        case ov::element::f16:
            cvt_f16_to_f32(h, src, dst);
            cvt_f32_to_i32(h, dst, dst, is_saturated);
            break;
        case ov::element::i8:
        case ov::element::u8:
            cvt_byte_to_i16(h, src, dst, input_type.is_signed());
            cvt_i16_to_i32(h, dst, dst);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
        }
        break;
    case ov::element::f16:
        switch (input_type) {
        case ov::element::f32:
            cvt_f32_to_f16(h, src, dst);
            break;
        case ov::element::i32:
            cvt_i32_to_f32(h, src, dst);
            cvt_f32_to_f16(h, dst, dst);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
        }
        break;
    case ov::element::i8:
    case ov::element::u8:
        switch (input_type) {
        case ov::element::f32:
            cvt_f32_to_i32(h, src, dst, is_saturated);
            cvt_i32_to_i16(h, dst, dst, is_saturated);
            cvt_i16_to_byte(h, dst, dst, output_type.is_signed(), is_saturated);
            break;
        case ov::element::i32:
            cvt_i32_to_i16(h, src, dst, is_saturated);
            cvt_i16_to_byte(h, dst, dst, output_type.is_signed(), is_saturated);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
        }
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported output type: ", output_type.get_type_name());
    }
}
}  // namespace

// Helper function to get max_offset and alignment for different register types
template <typename RegType>
static std::pair<int, int> get_load_store_limits() {
    int max_offset = 4095;
    int alignment = 1;

    if constexpr (std::is_same_v<RegType, VReg> || std::is_same_v<RegType, QReg>) {
        max_offset = 65520;  // 4095 * 16
        alignment = 16;
    } else if constexpr (std::is_same_v<RegType, DReg>) {
        max_offset = 32760;
        alignment = 8;
    } else if constexpr (std::is_same_v<RegType, SReg>) {
        max_offset = 16380;
        alignment = 4;
    } else if constexpr (std::is_same_v<RegType, HReg>) {
        max_offset = 8190;
        alignment = 2;
    } else if constexpr (std::is_same_v<RegType, BReg>) {
        max_offset = 4095;
        alignment = 1;
    }

    return {max_offset, alignment};
}

// Helper function to load with large offset handling
template <typename RegType>
static void load_with_offset_check(jit_generator* h, const RegType& dst, const XReg& src, int offset) {
    const auto [max_offset, alignment] = get_load_store_limits<RegType>();

    if (offset >= 0 && offset <= max_offset && (offset % alignment) == 0) {
        if constexpr (std::is_same_v<RegType, VReg> || std::is_same_v<RegType, QReg>) {
            // Create QReg from register index to ensure proper 128-bit SIMD register access
            // VReg is an alias for QReg, but explicit QReg construction guarantees correct instruction encoding
            h->ldr(QReg(dst.getIdx()), ptr(src, static_cast<uint32_t>(offset)));
        } else {
            h->ldr(dst, ptr(src, static_cast<uint32_t>(offset)));
        }
    } else {
        // Use add_imm which handles register allocation internally
        h->add_imm(h->X_DEFAULT_ADDR, src, offset, h->X_TMP_0);
        if constexpr (std::is_same_v<RegType, VReg> || std::is_same_v<RegType, QReg>) {
            // Create QReg from register index to ensure proper 128-bit SIMD register access
            h->ldr(QReg(dst.getIdx()), ptr(h->X_DEFAULT_ADDR));
        } else {
            h->ldr(dst, ptr(h->X_DEFAULT_ADDR));
        }
    }
}

// Helper function to store with large offset handling
template <typename RegType>
static void store_with_offset_check(jit_generator* h, const RegType& src, const XReg& dst, int offset) {
    const auto [max_offset, alignment] = get_load_store_limits<RegType>();

    if (offset >= 0 && offset <= max_offset && (offset % alignment) == 0) {
        if constexpr (std::is_same_v<RegType, VReg> || std::is_same_v<RegType, QReg>) {
            // Create QReg from register index to ensure proper 128-bit SIMD register access
            // VReg is an alias for QReg, but explicit QReg construction guarantees correct instruction encoding
            h->str(QReg(src.getIdx()), ptr(dst, static_cast<uint32_t>(offset)));
        } else {
            h->str(src, ptr(dst, static_cast<uint32_t>(offset)));
        }
    } else {
        // Use add_imm which handles register allocation internally
        h->add_imm(h->X_DEFAULT_ADDR, dst, offset, h->X_TMP_0);
        if constexpr (std::is_same_v<RegType, VReg> || std::is_same_v<RegType, QReg>) {
            // Create QReg from register index to ensure proper 128-bit SIMD register access
            h->str(QReg(src.getIdx()), ptr(h->X_DEFAULT_ADDR));
        } else {
            h->str(src, ptr(h->X_DEFAULT_ADDR));
        }
    }
}

jit_load_emitter::jit_load_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   ov::element::Type src_prc,
                                   ov::element::Type dst_prc,
                                   int load_num,
                                   int byte_offset,
                                   arithmetic_mode mode,
                                   ov::element::Type exec_prc,
                                   emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type),
      name_("unknown"),
      load_num_(load_num),
      byte_offset_(byte_offset),
      src_prc_(src_prc),
      dst_prc_(dst_prc),
      mode_(mode) {
    const bool is_supported_precision =
        any_of(src_prc_, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
        (src_prc_ == dst_prc_ || any_of(dst_prc_, ov::element::f32, ov::element::i32));
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");
}

void jit_load_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::load_qbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    auto dst_s = SReg(out_idxs[0]);
    auto dst_d = DReg(out_idxs[0]);

    switch (load_num_) {
    case 0:
        break;
    case 1:
        load_with_offset_check(h, dst_s, src, byte_offset_);
        break;
    case 2:
        load_with_offset_check(h, dst_d, src, byte_offset_);
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[0]);
        load_with_offset_check(h, dst_d, src, byte_offset_);
        h->add_imm(prc, src, byte_offset_ + 2 * sizeof(float), h->X_DEFAULT_ADDR);
        h->ld1(dst.s[2], ptr(prc));
        break;
    }
    case 4:
        load_with_offset_check(h, QReg(out_idxs[0]), src, byte_offset_);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to load.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::load_dbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    auto dst_h = HReg(out_idxs[0]);
    auto dst_s = SReg(out_idxs[0]);
    auto dst_d = DReg(out_idxs[0]);

    switch (load_num_) {
    case 0:
        break;
    case 1:
        load_with_offset_check(h, dst_h, src, byte_offset_);
        break;
    case 2:
        load_with_offset_check(h, dst_s, src, byte_offset_);
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[0]);
        load_with_offset_check(h, dst_s, src, byte_offset_);
        h->add_imm(prc, src, byte_offset_ + 2 * sizeof(uint16_t), h->X_DEFAULT_ADDR);
        h->ld1(dst.h[2], ptr(prc));
        break;
    }
    case 4:
        load_with_offset_check(h, dst_d, src, byte_offset_);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to load.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::load_byte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = XReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    auto dst_b = BReg(out_idxs[0]);
    auto dst_h = HReg(out_idxs[0]);
    auto dst_s = SReg(out_idxs[0]);

    switch (load_num_) {
    case 0:
        break;
    case 1:
        load_with_offset_check(h, dst_b, src, byte_offset_);
        break;
    case 2:
        load_with_offset_check(h, dst_h, src, byte_offset_);
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[0]);
        load_with_offset_check(h, dst_h, src, byte_offset_);
        h->add_imm(prc, src, byte_offset_ + 2 * sizeof(int8_t), h->X_DEFAULT_ADDR);
        h->ld1(dst.b[2], ptr(prc));
        break;
    }
    case 4:
        load_with_offset_check(h, dst_s, src, byte_offset_);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to load.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(
        any_of(src_prc_, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
        "Unsupported precision.");
    OV_CPU_JIT_EMITTER_ASSERT(load_num_ <= 4, "Unexpected number of elements to load.");

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const auto dst = TReg(out_idxs[0]);

    switch (src_prc_) {
    case ov::element::f32:
    case ov::element::i32:
        load_qbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::f16:
        load_dbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::i8:
    case ov::element::u8:
        load_byte<isa>(in_idxs, out_idxs);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", src_prc_.get_type_name());
    }

    if (src_prc_ != dst_prc_) {
        jit_convert_process(h, dst, dst, src_prc_, dst_prc_, mode_ == arithmetic_mode::saturation);
    }
}

size_t jit_load_emitter::get_aux_gprs_count() const {
    if (load_num_ == 3) {
        return 1;
    }

    return 0;
}

jit_store_emitter::jit_store_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     ov::element::Type src_prc,
                                     ov::element::Type dst_prc,
                                     int store_num,
                                     int byte_offset,
                                     arithmetic_mode mode,
                                     ov::element::Type exec_prc,
                                     emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type),
      name_("unknown"),
      store_num_(store_num),
      byte_offset_(byte_offset),
      src_prc_(src_prc),
      dst_prc_(dst_prc),
      mode_(mode) {
    const bool is_supported_precision =
        any_of(dst_prc_, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8) &&
        (src_prc_ == dst_prc_ || any_of(src_prc_, ov::element::f32, ov::element::i32));
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_precision, "Unsupported precision pair.");
}

void jit_store_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::store_qbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto src_s = SReg(in_idxs[0]);
    auto src_d = DReg(in_idxs[0]);
    auto src_q = QReg(in_idxs[0]);
    auto dst = XReg(out_idxs[0]);

    switch (store_num_) {
    case 0:
        break;
    case 1:
        store_with_offset_check(h, src_s, dst, byte_offset_);
        break;
    case 2:
        store_with_offset_check(h, src_d, dst, byte_offset_);
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[0]);
        store_with_offset_check(h, src_d, dst, byte_offset_);
        h->add_imm(prc, dst, byte_offset_ + 2 * sizeof(float), h->X_DEFAULT_ADDR);
        h->st1(src.s[2], ptr(prc));
        break;
    }
    case 4:
        store_with_offset_check(h, src_q, dst, byte_offset_);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to store.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::store_dbyte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto src_h = HReg(in_idxs[0]);
    auto src_s = SReg(in_idxs[0]);
    auto src_d = DReg(in_idxs[0]);
    auto dst = XReg(out_idxs[0]);

    switch (store_num_) {
    case 0:
        break;
    case 1:
        store_with_offset_check(h, src_h, dst, byte_offset_);
        break;
    case 2:
        store_with_offset_check(h, src_s, dst, byte_offset_);
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[0]);
        store_with_offset_check(h, src_s, dst, byte_offset_);
        h->add_imm(prc, dst, byte_offset_ + 2 * sizeof(uint16_t), h->X_DEFAULT_ADDR);
        h->st1(src.h[2], ptr(prc));
        break;
    }
    case 4:
        store_with_offset_check(h, src_d, dst, byte_offset_);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to store.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::store_byte(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto src_b = BReg(in_idxs[0]);
    auto src_h = HReg(in_idxs[0]);
    auto src_s = SReg(in_idxs[0]);
    auto dst = XReg(out_idxs[0]);

    switch (store_num_) {
    case 0:
        break;
    case 1:
        store_with_offset_check(h, src_b, dst, byte_offset_);
        break;
    case 2:
        store_with_offset_check(h, src_h, dst, byte_offset_);
        break;
    case 3: {
        auto prc = XReg(aux_gpr_idxs[0]);
        store_with_offset_check(h, src_h, dst, byte_offset_);
        h->add_imm(prc, dst, byte_offset_ + 2 * sizeof(int8_t), h->X_DEFAULT_ADDR);
        h->st1(src.b[2], ptr(prc));
        break;
    }
    case 4:
        store_with_offset_check(h, src_s, dst, byte_offset_);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unexpected number of elements to store.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(
        any_of(dst_prc_, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
        "Unsupported precision.");
    OV_CPU_JIT_EMITTER_ASSERT(store_num_ <= 4, "Unexpected number of elements to store.");

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const auto src = TReg(in_idxs[0]);
    if (src_prc_ != dst_prc_) {
        jit_convert_process(h, src, src, src_prc_, dst_prc_, mode_ == arithmetic_mode::saturation);
    }

    switch (dst_prc_) {
    case ov::element::f32:
    case ov::element::i32:
        store_qbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::f16:
        store_dbyte<isa>(in_idxs, out_idxs);
        break;
    case ov::element::i8:
    case ov::element::u8:
        store_byte<isa>(in_idxs, out_idxs);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", dst_prc_.get_type_name());
    }
}

size_t jit_store_emitter::get_aux_gprs_count() const {
    if (store_num_ == 3) {
        return 1;
    }

    return 0;
}

}  // namespace ov::intel_cpu::aarch64
