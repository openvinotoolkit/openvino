// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/jit_generator.hpp>

#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64::jit_conversion {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator_t;

// In aarch64, conversion between f16 and i16/u16 can be done with single instruction. The supported
// conversion precisions are f32, i32, f16, i8 (byte), u8 (byte). If we introduce an intermediate
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
inline void cvt_f16_to_f32(jit_generator* h, const TReg& src, const TReg& dst) {
    h->fcvtl(dst.s4, src.h4);
}

template <typename TReg>
inline void cvt_f32_to_f16(jit_generator* h, const TReg& src, const TReg& dst) {
    h->fcvtn(dst.h4, src.s4);
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
inline void cvt_f16_to_i16(jit_generator* h, const TReg& src, const TReg& dst) {
    h->fcvtzs(dst.h4, src.h4);
}

template <typename TReg>
inline void cvt_i16_to_f16(jit_generator* h, const TReg& src, const TReg& dst) {
    h->scvtf(dst.h4, src.h4);
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
inline void cvt_byte_to_i16(jit_generator* h, const TReg& src, const TReg& dst, bool is_signed) {
    if (is_signed) {
        h->sxtl(dst.h8, src.b8);
    } else {
        h->uxtl(dst.h8, src.b8);
    }
}

template <typename TReg>
void emit_convert_process(jit_generator* h,
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
        case ov::element::i8:
        case ov::element::u8:
            cvt_byte_to_i16(h, src, dst, input_type.is_signed());
            cvt_i16_to_i32(h, dst, dst);
            cvt_i32_to_f32(h, dst, dst);
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
        case ov::element::f16:
            cvt_f16_to_f32(h, src, dst);
            cvt_f32_to_i32(h, dst, dst, is_saturated);
            cvt_i32_to_i16(h, dst, dst, is_saturated);
            cvt_i16_to_byte(h, dst, dst, output_type.is_signed(), is_saturated);
            break;
        case ov::element::i8:
        case ov::element::u8:
            cvt_byte_to_i16(h, src, dst, input_type.is_signed());
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

}  // namespace ov::intel_cpu::aarch64::jit_conversion
