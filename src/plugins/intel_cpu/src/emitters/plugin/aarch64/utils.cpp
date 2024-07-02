// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

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
void cvt_i32_to_byte(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                     bool is_signed, bool is_saturated) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_saturated) {
        if (is_signed) {
            h->sqxtn(dst.h4, src.s4);
            h->sqxtn(dst.b8, dst.h8);
        } else {
            h->uqxtn(dst.h4, src.s4);
            h->uqxtn(dst.b8, dst.h8);
        }
    } else {
        h->xtn(dst.h4, src.s4);
        h->xtn(dst.b8, dst.h8);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void cvt_byte_to_i32(dnnl::impl::cpu::aarch64::jit_generator* h, const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                     bool is_signed) {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_signed) {
        h->sxtl(dst.h8, src.b8);
        h->sxtl(dst.s4, dst.h4);
    } else {
        h->uxtl(dst.h8, src.b8);
        h->uxtl(dst.s4, dst.h4);
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

template void cvt_i32_to_byte<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                              const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                              bool is_signed, bool is_saturation);

template void cvt_byte_to_i32<dnnl::impl::cpu::aarch64::asimd>(dnnl::impl::cpu::aarch64::jit_generator* h,
                              const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                              bool is_signed);

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
