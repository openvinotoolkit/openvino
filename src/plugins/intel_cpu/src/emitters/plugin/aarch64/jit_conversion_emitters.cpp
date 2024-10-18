// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"
#include "emitters/utils.hpp"

using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

// In aarch64, conversion between f16 and i16/u16 can be done with single instruction. The supported
// conversion precicions are f32, i32, f16, i8 (byte), u8 (byte). If we introduce an intermediate
// precision i16 in the following graph. Then the conversion between each pair of
// neighbors in this graph will be done with single instruction.
// f16 - f32 - i32 - i16 - byte
//  |                 |
//  - - - - - - - - - -
// Note that using single instruction for conversion between f16 and i16 is only available for
// architecture ARMv8.2-A or later versions. So ARM platforms like Raspberry (Model name Cortex-A72)
// with architecture ARMv8 do not support such instructions. And as the isa asimd we supported
// does not distinguish ARMv8.2 with ARMv8.2-A, conversion between f16 and i16 will still use three
// instructions f16 -> f32 -> i32 -> i16 (f16 <- f32 <- i32 <- i16).
template <typename TReg>
inline void jit_convert_emitter::cvt_f16_to_f32(const TReg &src, const TReg &dst) const {
    h->fcvtl(dst.s4, src.h4);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_f32_to_f16(const TReg &src, const TReg &dst) const {
    h->fcvtn(dst.h4, src.s4);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_f32_to_i32(const TReg &src, const TReg &dst) const {
    h->fcvtzs(dst.s, src.s);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_i32_to_f32(const TReg &src, const TReg &dst) const {
    h->scvtf(dst.s, src.s);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_i32_to_i16(const TReg &src, const TReg &dst, bool is_saturated) const {
    if (is_saturated) {
        h->sqxtn(dst.h4, src.s4);
    } else {
        h->xtn(dst.h4, src.s4);
    }
}

template <typename TReg>
inline void jit_convert_emitter::cvt_i16_to_i32(const TReg &src, const TReg &dst) const {
    h->sxtl(dst.s4, src.h4);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_f16_to_i16(const TReg &src, const TReg &dst) const {
    h->fcvtzs(dst.h4, src.h4);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_i16_to_f16(const TReg &src, const TReg &dst) const {
    h->scvtf(dst.h4, src.h4);
}

template <typename TReg>
inline void jit_convert_emitter::cvt_i16_to_byte(const TReg &src, const TReg &dst, bool is_signed, bool is_saturated) const {
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
inline void jit_convert_emitter::cvt_byte_to_i16(const TReg &src, const TReg &dst, bool is_signed) const {
    if (is_signed) {
        h->sxtl(dst.h8, src.b8);
    } else {
        h->uxtl(dst.h8, src.b8);
    }
}

template <typename TReg>
void jit_convert_emitter::jit_convert_process(const TReg &src, const TReg &dst, ov::element::Type input_type, ov::element::Type output_type,
                                              bool is_saturated) const {
    if (input_type == output_type || (!is_saturated &&
        one_of(input_type, ov::element::i8, ov::element::u8) && one_of(output_type, ov::element::i8, ov::element::u8))) {
        if (src.getIdx() != dst.getIdx()) {
            h->mov(dst.b16, src.b16);
        }
        return;
    }

    switch (output_type) {
        case ov::element::f32:
            switch (input_type) {
                case ov::element::i32:
                    cvt_i32_to_f32<TReg>(src, dst);
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<TReg>(src, dst);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i16<TReg>(src, dst, input_type.is_signed());
                    cvt_i16_to_i32<TReg>(dst, dst);
                    cvt_i32_to_f32<TReg>(dst, dst);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::i32:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_i32<TReg>(src, dst);
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<TReg>(src, dst);
                    cvt_f32_to_i32<TReg>(dst, dst);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i16<TReg>(src, dst, input_type.is_signed());
                    cvt_i16_to_i32<TReg>(dst, dst);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::f16:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_f16<TReg>(src, dst);
                    break;
                case ov::element::i32:
                    cvt_i32_to_f32<TReg>(src, dst);
                    cvt_f32_to_f16<TReg>(dst, dst);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i16<TReg>(src, dst, input_type.is_signed());
                    cvt_i16_to_i32<TReg>(dst, dst);
                    cvt_i32_to_f32<TReg>(dst, dst);
                    cvt_f32_to_f16<TReg>(dst, dst);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::i8:
        case ov::element::u8:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_i32<TReg>(src, dst);
                    cvt_i32_to_i16<TReg>(dst, dst, is_saturated);
                    cvt_i16_to_byte<TReg>(dst, dst, output_type.is_signed(), is_saturated);
                    break;
                case ov::element::i32:
                    cvt_i32_to_i16<TReg>(src, dst, is_saturated);
                    cvt_i16_to_byte<TReg>(dst, dst, output_type.is_signed(), is_saturated);
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<TReg>(src, dst);
                    cvt_f32_to_i32<TReg>(dst, dst);
                    cvt_i32_to_i16<TReg>(dst, dst, is_saturated);
                    cvt_i16_to_byte<TReg>(dst, dst, output_type.is_signed(), is_saturated);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i16<TReg>(src, dst, input_type.is_signed());
                    cvt_i16_to_byte<TReg>(dst, dst, output_type.is_signed(), is_saturated);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported output type: ", output_type.get_type_name());
    }
}

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    input_type = node->get_input_element_type(0);
    output_type = node->get_output_element_type(0);
}

void jit_convert_emitter::validate_types() const {
    OV_CPU_JIT_EMITTER_ASSERT(one_of(input_type, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
                              "Unsupported input type: ", input_type.get_type_name());
    OV_CPU_JIT_EMITTER_ASSERT(one_of(output_type, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
                              "Unsupported output type: ", output_type.get_type_name());
}

size_t jit_convert_emitter::get_inputs_count() const { return 1; }

void jit_convert_emitter::emit_data() const {
    jit_emitter::emit_data();
}

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

void jit_convert_truncation_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_truncation_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    jit_convert_process<TReg>(src, dst, input_type, output_type, false);
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

void jit_convert_saturation_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_saturation_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    jit_convert_process<TReg>(src, dst, input_type, output_type, true);
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
