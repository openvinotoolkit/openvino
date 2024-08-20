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
// precision i16/u16 (dbyte) in the following graph. Then the conversion between each pair of
// neighbors in this graph will be done with single instruction.
// f16 - f32 - i32 - dbyte - byte
//  |                   |
//  - - - - - - - - - - -
template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_f16_to_f32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtl(dst.s4, src.h4);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_f32_to_f16(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtn(dst.h4, src.s4);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_f32_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtzs(dst.s, src.s);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_i32_to_f32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->scvtf(dst.s, src.s);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_i32_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                     bool is_signed, bool is_saturated) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_saturated) {
        if (is_signed) {
            h->sqxtn(dst.h4, src.s4);
        } else {
            h->uqxtn(dst.h4, src.s4);
        }
    } else {
        h->xtn(dst.h4, src.s4);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_dbyte_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                     bool is_signed) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_signed) {
        h->sxtl(dst.s4, src.h4);
    } else {
        h->uxtl(dst.s4, src.h4);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_f16_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtzs(dst.h4, src.h4);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_dbyte_to_f16(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  bool is_signed) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_signed) {
        h->scvtf(dst.h4, src.h4);
    } else {
        h->ucvtf(dst.h4, src.h4);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_dbyte_to_byte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                     bool is_signed, bool is_saturated) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
static void cvt_byte_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  bool is_signed) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_signed) {
        h->sxtl(dst.h8, src.b8);
    } else {
        h->uxtl(dst.h8, src.b8);
    }
}

template <cpu_isa_t isa>
static void jit_convert_process(dnnl::impl::cpu::aarch64::jit_generator* h,
                                const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                                ov::element::Type input_type, ov::element::Type output_type, bool is_saturated) {
    if (input_type == output_type) {
        using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
        h->mov(TReg(out_idxs[0]).b16, TReg(in_idxs[0]).b16);
        return;
    }

    switch (output_type) {
        case ov::element::f32:
            switch (input_type) {
                case ov::element::f32:
                    break;
                case ov::element::i32:
                    cvt_i32_to_f32<isa>(h, in_idxs, out_idxs);
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<isa>(h, in_idxs, out_idxs);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_dbyte<isa>(h, in_idxs, out_idxs, input_type.is_signed());
                    cvt_dbyte_to_i32<isa>(h, out_idxs, out_idxs, input_type.is_signed());
                    cvt_i32_to_f32<isa>(h, out_idxs, out_idxs);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::i32:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_i32<isa>(h, in_idxs, out_idxs);
                    break;
                case ov::element::i32:
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<isa>(h, in_idxs, out_idxs);
                    cvt_f32_to_i32<isa>(h, out_idxs, out_idxs);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_dbyte<isa>(h, in_idxs, out_idxs, input_type.is_signed());
                    cvt_dbyte_to_i32<isa>(h, out_idxs, out_idxs, input_type.is_signed());
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::f16:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_f16<isa>(h, in_idxs, out_idxs);
                    break;
                case ov::element::i32:
                    cvt_i32_to_f32<isa>(h, in_idxs, out_idxs);
                    cvt_f32_to_f16<isa>(h, out_idxs, out_idxs);
                    break;
                case ov::element::f16:
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_dbyte<isa>(h, in_idxs, out_idxs, input_type.is_signed());
                    cvt_dbyte_to_f16<isa>(h, out_idxs, out_idxs, input_type.is_signed());
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::i8:
        case ov::element::u8:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_i32<isa>(h, in_idxs, out_idxs);
                    cvt_i32_to_dbyte<isa>(h, out_idxs, out_idxs, output_type.is_signed(), is_saturated);
                    cvt_dbyte_to_byte<isa>(h, out_idxs, out_idxs, output_type.is_signed(), is_saturated);
                    break;
                case ov::element::i32:
                    cvt_i32_to_dbyte<isa>(h, in_idxs, out_idxs, output_type.is_signed(), is_saturated);
                    cvt_dbyte_to_byte<isa>(h, out_idxs, out_idxs, output_type.is_signed(), is_saturated);
                    break;
                case ov::element::f16:
                    cvt_f16_to_dbyte<isa>(h, in_idxs, out_idxs);
                    cvt_dbyte_to_byte<isa>(h, out_idxs, out_idxs, output_type.is_signed(), is_saturated);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_dbyte<isa>(h, in_idxs, out_idxs, input_type.is_signed());
                    cvt_dbyte_to_byte<isa>(h, out_idxs, out_idxs, output_type.is_signed(), is_saturated);
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

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa,
                                         ov::element::Type input_prc,
                                         ov::element::Type output_prc,
                                         ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    input_type = input_prc;
    output_type = output_prc;
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

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               ov::element::Type input_prc,
                                                               ov::element::Type output_prc,
                                                               ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, input_prc, output_prc, exec_prc) {
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
    jit_convert_process<isa>(h, in_idxs, out_idxs, input_type, output_type, false);
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               ov::element::Type input_prc,
                                                               ov::element::Type output_prc,
                                                               ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, input_prc, output_prc, exec_prc) {
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
    jit_convert_process<isa>(h, in_idxs, out_idxs, input_type, output_type, true);
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
