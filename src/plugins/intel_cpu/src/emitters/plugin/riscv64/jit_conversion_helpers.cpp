// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_helpers.hpp"

#include <cstddef>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <utility>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64::jit_conversion {
namespace {

std::pair<Xbyak_riscv::SEW, Xbyak_riscv::LMUL> get_vtype_for_element_size(const ov::element::Type& type) {
    switch (type.size()) {
    case 4UL:
        return {Xbyak_riscv::SEW::e32, Xbyak_riscv::LMUL::m1};
    case 2UL:
        return {Xbyak_riscv::SEW::e16, Xbyak_riscv::LMUL::mf2};
    case 1UL:
        return {Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::mf4};
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision size for vtype setup: ", type);
    }
}

bool requires_zvfh(const ov::element::Type& input_type, const ov::element::Type& output_type) {
    return input_type != output_type && any_of(ov::element::f16, input_type, output_type);
}

}  // namespace

bool is_supported_convert_precision(const ov::element::Type& precision) {
    return any_of(precision, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8);
}

void validate_convert_precision(const ov::element::Type& input_type, const ov::element::Type& output_type) {
    OV_CPU_JIT_EMITTER_ASSERT(is_supported_convert_precision(input_type) && is_supported_convert_precision(output_type),
                              "Unsupported conversion: ",
                              input_type,
                              " -> ",
                              output_type);
    OV_CPU_JIT_EMITTER_ASSERT(
        !requires_zvfh(input_type, output_type) || mayiuse(ov::intel_cpu::riscv64::cpu_isa_t::gv_zvfh),
        "Unsupported Zvfh conversion: ",
        input_type,
        " -> ",
        output_type);
}

Xbyak_riscv::SEW byte_size_to_sew(size_t byte_size) {
    switch (byte_size) {
    case 1UL:
        return Xbyak_riscv::SEW::e8;
    case 2UL:
        return Xbyak_riscv::SEW::e16;
    case 4UL:
        return Xbyak_riscv::SEW::e32;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported memory access byte size: ", byte_size);
    }
}

void emit_convert_process(ov::intel_cpu::riscv64::jit_generator_t* h,
                          const Xbyak_riscv::VReg& src,
                          const Xbyak_riscv::VReg& dst,
                          const ov::element::Type& input_type,
                          const ov::element::Type& output_type,
                          arithmetic_mode mode,
                          const Xbyak_riscv::Reg& avl) {
    const auto is_saturation = mode == arithmetic_mode::saturation;
    h->csrr(avl, Xbyak_riscv::CSR::vl);

    const auto set_type_for = [&](const ov::element::Type& type) {
        const auto [sew, lmul] = get_vtype_for_element_size(type);
        set_vector_length(h, 0, sew, {}, lmul, &avl);
    };
    const auto move_if_needed = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to) {
        if (from.getIdx() != to.getIdx()) {
            h->vmv_v_v(to, from);
        }
    };
    const auto narrow_i32_to_i8 = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to) {
        set_type_for(ov::element::i16);
        if (is_saturation) {
            h->vnclip_wi(to, from, 0);
        } else {
            h->vnsra_wi(to, from, 0);
        }
        set_type_for(ov::element::i8);
        if (is_saturation) {
            h->vnclip_wi(to, to, 0);
        } else {
            h->vnsra_wi(to, to, 0);
        }
    };
    const auto narrow_i32_to_u8 = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to) {
        set_type_for(ov::element::i16);
        if (is_saturation) {
            h->vnclipu_wi(to, from, 0);
        } else {
            h->vnsrl_wi(to, from, 0);
        }
        set_type_for(ov::element::i8);
        if (is_saturation) {
            h->vnclipu_wi(to, to, 0);
        } else {
            h->vnsrl_wi(to, to, 0);
        }
    };
    const auto widen_f16_to_f32 = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to) {
        set_type_for(ov::element::f16);
        h->vfwcvt_f_f_v(to, from);
        set_type_for(ov::element::f32);
    };
    const auto narrow_f32_to_f16 = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to) {
        set_type_for(ov::element::f16);
        h->vfncvt_f_f_w(to, from);
    };
    const auto widen_i8_to_i32 = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to, const bool is_signed) {
        set_type_for(ov::element::i32);
        if (is_signed) {
            h->vsext_vf4(to, from);
        } else {
            h->vzext_vf4(to, from);
        }
    };
    const auto saturate_i32_to_u8 = [&](const Xbyak_riscv::VReg& from, const Xbyak_riscv::VReg& to) {
        set_type_for(ov::element::i32);
        move_if_needed(from, to);
        h->vfcvt_f_x_v(to, to);
        h->vfcvt_xu_f_v(to, to);
        narrow_i32_to_u8(to, to);
    };

    if (input_type == output_type) {
        set_type_for(input_type);
        move_if_needed(src, dst);
        return;
    }

    if (output_type == ov::element::f32) {
        if (input_type == ov::element::f16) {
            widen_f16_to_f32(src, dst);
        } else if (input_type == ov::element::i32) {
            set_type_for(ov::element::i32);
            h->vfcvt_f_x_v(dst, src);
            set_type_for(ov::element::f32);
        } else if (input_type == ov::element::i8) {
            widen_i8_to_i32(src, dst, true);
            h->vfcvt_f_x_v(dst, dst);
            set_type_for(ov::element::f32);
        } else if (input_type == ov::element::u8) {
            widen_i8_to_i32(src, dst, false);
            h->vfcvt_f_xu_v(dst, dst);
            set_type_for(ov::element::f32);
        } else {
            OV_CPU_JIT_EMITTER_THROW("Unsupported conversion: ", input_type, " -> ", output_type);
        }
        return;
    }

    if (output_type == ov::element::f16) {
        if (input_type == ov::element::f32) {
            narrow_f32_to_f16(src, dst);
        } else if (input_type == ov::element::i32) {
            set_type_for(ov::element::i32);
            h->vfcvt_f_x_v(dst, src);
            if (is_saturation) {
                set_type_for(ov::element::f32);
            }
            narrow_f32_to_f16(dst, dst);
        } else if (input_type == ov::element::i8) {
            widen_i8_to_i32(src, dst, true);
            h->vfcvt_f_x_v(dst, dst);
            if (is_saturation) {
                set_type_for(ov::element::f32);
            }
            narrow_f32_to_f16(dst, dst);
        } else if (input_type == ov::element::u8) {
            widen_i8_to_i32(src, dst, false);
            h->vfcvt_f_xu_v(dst, dst);
            if (is_saturation) {
                set_type_for(ov::element::f32);
            }
            narrow_f32_to_f16(dst, dst);
        } else {
            OV_CPU_JIT_EMITTER_THROW("Unsupported conversion: ", input_type, " -> ", output_type);
        }
        return;
    }

    if (output_type == ov::element::i32) {
        if (input_type == ov::element::f32) {
            set_type_for(ov::element::f32);
            if (is_saturation) {
                h->vfcvt_x_f_v(dst, src);
            } else {
                h->vfcvt_rtz_x_f_v(dst, src);
            }
            set_type_for(ov::element::i32);
        } else if (input_type == ov::element::f16) {
            widen_f16_to_f32(src, dst);
            if (is_saturation) {
                h->vfcvt_x_f_v(dst, dst);
            } else {
                h->vfcvt_rtz_x_f_v(dst, dst);
            }
            set_type_for(ov::element::i32);
        } else if (any_of(input_type, ov::element::i8, ov::element::u8)) {
            widen_i8_to_i32(src, dst, input_type == ov::element::i8);
        } else {
            OV_CPU_JIT_EMITTER_THROW("Unsupported conversion: ", input_type, " -> ", output_type);
        }
        return;
    }

    if (output_type == ov::element::i8) {
        if (input_type == ov::element::f32) {
            set_type_for(ov::element::f32);
            if (is_saturation) {
                h->vfcvt_x_f_v(dst, src);
            } else {
                h->vfcvt_rtz_x_f_v(dst, src);
            }
            set_type_for(ov::element::i32);
            narrow_i32_to_i8(dst, dst);
        } else if (input_type == ov::element::f16) {
            widen_f16_to_f32(src, dst);
            if (is_saturation) {
                h->vfcvt_x_f_v(dst, dst);
            } else {
                h->vfcvt_rtz_x_f_v(dst, dst);
            }
            set_type_for(ov::element::i32);
            narrow_i32_to_i8(dst, dst);
        } else if (input_type == ov::element::i32) {
            narrow_i32_to_i8(src, dst);
        } else if (input_type == ov::element::u8) {
            if (is_saturation) {
                widen_i8_to_i32(src, dst, false);
                narrow_i32_to_i8(dst, dst);
            } else {
                set_type_for(ov::element::i8);
                move_if_needed(src, dst);
            }
        } else {
            OV_CPU_JIT_EMITTER_THROW("Unsupported conversion: ", input_type, " -> ", output_type);
        }
        return;
    }

    if (output_type == ov::element::u8) {
        if (input_type == ov::element::f32) {
            set_type_for(ov::element::f32);
            if (is_saturation) {
                h->vfcvt_xu_f_v(dst, src);
            } else {
                h->vfcvt_rtz_x_f_v(dst, src);
            }
            set_type_for(ov::element::i32);
            narrow_i32_to_u8(dst, dst);
        } else if (input_type == ov::element::f16) {
            widen_f16_to_f32(src, dst);
            if (is_saturation) {
                h->vfcvt_xu_f_v(dst, dst);
            } else {
                h->vfcvt_rtz_x_f_v(dst, dst);
            }
            set_type_for(ov::element::i32);
            narrow_i32_to_u8(dst, dst);
        } else if (input_type == ov::element::i32) {
            if (is_saturation) {
                saturate_i32_to_u8(src, dst);
            } else {
                narrow_i32_to_u8(src, dst);
            }
        } else if (input_type == ov::element::i8) {
            if (is_saturation) {
                widen_i8_to_i32(src, dst, true);
                saturate_i32_to_u8(dst, dst);
            } else {
                set_type_for(ov::element::i8);
                move_if_needed(src, dst);
            }
        } else {
            OV_CPU_JIT_EMITTER_THROW("Unsupported conversion: ", input_type, " -> ", output_type);
        }
        return;
    }

    OV_CPU_JIT_EMITTER_THROW("Unsupported conversion: ", input_type, " -> ", output_type);
}

}  // namespace ov::intel_cpu::riscv64::jit_conversion
