// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_load_store_emitters.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;

jit_load_emitter::jit_load_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   ov::element::Type src_prc, ov::element::Type dst_prc, int load_num, int byte_offset,
                                   ov::element::Type exec_prc, emitter_in_out_map in_out_type)
: jit_emitter(host, host_isa, exec_prc, in_out_type), name_("unknown"), load_num_(load_num), byte_offset_(byte_offset),
              src_prc_(src_prc), dst_prc_(dst_prc) {
    prepare_table();
}

void jit_load_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OPENVINO_THROW("Load emitter in ", name_, " is performed on unsupported isa.");
    }
}

template <cpu_isa_t isa>
void jit_load_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    bool matched_prc = src_prc_ == ov::element::f32 && dst_prc_ == ov::element::f32;
    if (!matched_prc)
        OPENVINO_THROW("Load emitter in ", name_, " only support both input and output precisions of being FP32.");
    if (load_num_ > static_cast<int>((get_vec_length() / dst_prc_.size())))
        OPENVINO_THROW("Load emitter in ", name_, " have unexpected number of elements to load.");

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    XReg src = XReg(in_idxs[0]);
    XReg prc = XReg(aux_gpr_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    SReg dst_s = SReg(out_idxs[0]);
    DReg dst_d = DReg(out_idxs[0]);

    h->add(prc, src, byte_offset_);

    switch (load_num_) {
        case 0:
            break;
        case 1:
            h->ldr(dst_s, ptr(prc));
            break;
        case 2:
            h->ldr(dst_d, ptr(prc));
            break;
        case 3:
            h->ldr(dst_d, ptr(prc));
            h->add(prc, prc, 2 * sizeof(float));
            h->ld1(dst.s[2], ptr(prc));
            break;
        case 4:
            h->uni_ldr(dst, prc);
            break;
        default:
            OPENVINO_THROW("Load emitter in ", name_, " has unexpected number of elements to load.");
    }
}

size_t jit_load_emitter::get_aux_gprs_count() const {
    return 1;
}

jit_store_emitter::jit_store_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     ov::element::Type src_prc, ov::element::Type dst_prc, int store_num, int byte_offset,
                                     ov::element::Type exec_prc, emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type), name_("unknown"), store_num_(store_num), byte_offset_(byte_offset),
                  src_prc_(src_prc), dst_prc_(dst_prc) {
    prepare_table();
}

void jit_store_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OPENVINO_THROW("Store emitter in ", name_, " is performed on unsupported isa.");
    }
}

template <cpu_isa_t isa>
void jit_store_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    bool matched_prc = src_prc_ == ov::element::f32 && dst_prc_ == ov::element::f32;
    if (!matched_prc)
        OPENVINO_THROW("Store emitter in ", name_, " only support both input and output precisions of being FP32.");
    if (store_num_ > static_cast<int>((get_vec_length() / dst_prc_.size())))
        OPENVINO_THROW("Store emitter in ", name_, " have unexpected number of elements to store.");

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    SReg src_s = SReg(in_idxs[0]);
    DReg src_d = DReg(in_idxs[0]);
    XReg dst = XReg(out_idxs[0]);
    XReg prc = XReg(aux_gpr_idxs[0]);

    h->add(prc, dst, byte_offset_);

    switch (store_num_) {
        case 0:
            break;
        case 1:
            h->str(src_s, ptr(prc));
            break;
        case 2:
            h->str(src_d, ptr(prc));
            break;
        case 3:
            h->str(src_d, ptr(prc));
            h->add(prc, prc, 2 * sizeof(float));
            h->st1(src.s[2], ptr(prc));
            break;
        case 4:
            h->str(QReg(src.getIdx()), ptr(prc));
            break;
        default:
            OPENVINO_THROW("Store emitter in ", name_, " has unexpected number of elements to store.");
    }
}

size_t jit_store_emitter::get_aux_gprs_count() const {
    return 1;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
