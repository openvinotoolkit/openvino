// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

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
void cvt_f16_to_f32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtl(dst.s4, src.h4);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_f32_to_f16(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtn(dst.h4, src.s4);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_f32_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtzs(dst.s, src.s);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_i32_to_f32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->scvtf(dst.s, src.s);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_i32_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
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
void cvt_dbyte_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
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
void cvt_f16_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtzs(dst.h, src.h);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_dbyte_to_f16(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  bool is_signed) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_signed) {
        h->scvtf(dst.h, src.h);
    } else {
        h->ucvtf(dst.h, src.h);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_dbyte_to_byte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
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
void cvt_byte_to_dbyte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
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

template void cvt_f16_to_f32<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template void cvt_f32_to_f16<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template void cvt_f32_to_i32<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template void cvt_i32_to_f32<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template void cvt_i32_to_dbyte<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                             bool is_signed, bool is_saturation);

template void cvt_dbyte_to_i32<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                             bool is_signed);

template void cvt_f16_to_dbyte<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs);

template void cvt_dbyte_to_f16<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                             bool is_signed);

template void cvt_dbyte_to_byte<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                             bool is_signed, bool is_saturation);

template void cvt_byte_to_dbyte<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                             const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                             bool is_signed);

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
