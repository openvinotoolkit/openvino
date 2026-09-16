// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_load_store_emitters.hpp"

#include <cstddef>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <vector>

#include "emitters/plugin/riscv64/jit_conversion_helpers.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {
namespace {

bool needs_aux_vec_for_conversion(const ov::element::Type& src_prc, const ov::element::Type& dst_prc) {
    return src_prc != dst_prc && src_prc.size() < dst_prc.size();
}

bool needs_conversion(const ov::element::Type& src_prc, const ov::element::Type& dst_prc) {
    return src_prc != dst_prc;
}

}  // namespace

jit_load_emitter::jit_load_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   ov::element::Type src_prc,
                                   ov::element::Type dst_prc,
                                   size_t load_num,
                                   size_t byte_offset,
                                   arithmetic_mode mode,
                                   ov::element::Type exec_prc,
                                   emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type),
      load_num_(load_num),
      byte_offset_(byte_offset),
      src_prc_(src_prc),
      dst_prc_(dst_prc),
      mode_(mode) {
    jit_conversion::validate_convert_precision(src_prc_, dst_prc_);
}

size_t jit_load_emitter::get_inputs_num() const {
    return 1;
}

void jit_load_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa.");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_load_emitter::emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    auto src = Xbyak_riscv::Reg(in_idxs[0]);
    auto dst = Xbyak_riscv::VReg(out_idxs[0]);
    const auto needs_aux_vec = needs_aux_vec_for_conversion(src_prc_, dst_prc_);
    OV_CPU_JIT_EMITTER_ASSERT(!needs_aux_vec || !aux_vec_idxs.empty(),
                              "Widening load conversion requires an auxiliary vector register");
    auto data = needs_aux_vec ? Xbyak_riscv::VReg(aux_vec_idxs.front()) : dst;

    const auto byte_size = src_prc_.size();
    set_vector_length(h, load_num_, jit_conversion::byte_size_to_sew(byte_size), aux_gpr_idxs);

    if (byte_offset_ == 0) {
        if (byte_size == 1) {
            h->vle8_v(data, src);
        } else if (byte_size == 2) {
            h->vle16_v(data, src);
        } else {
            h->vle32_v(data, src);
        }
    } else {
        OV_CPU_JIT_EMITTER_ASSERT(!aux_gpr_idxs.empty(), "Static byte offset requires an auxiliary GPR");
        auto tmp_gpr = Xbyak_riscv::Reg(aux_gpr_idxs.front());
        h->uni_li(tmp_gpr, byte_offset_);
        h->add(tmp_gpr, src, tmp_gpr);
        if (byte_size == 1) {
            h->vle8_v(data, tmp_gpr);
        } else if (byte_size == 2) {
            h->vle16_v(data, tmp_gpr);
        } else {
            h->vle32_v(data, tmp_gpr);
        }
    }

    if (needs_conversion(src_prc_, dst_prc_)) {
        OV_CPU_JIT_EMITTER_ASSERT(!aux_gpr_idxs.empty(), "Load conversion requires an auxiliary GPR");
        const auto avl = Xbyak_riscv::Reg(aux_gpr_idxs.front());
        jit_conversion::emit_convert_process(h, data, dst, src_prc_, dst_prc_, mode_, avl);
    }
}

size_t jit_load_emitter::aux_gprs_count() const {
    return (load_num_ > 31 || byte_offset_ != 0 || needs_conversion(src_prc_, dst_prc_)) ? 1 : 0;
}

size_t jit_load_emitter::aux_vecs_count() const {
    return needs_aux_vec_for_conversion(src_prc_, dst_prc_) ? 1 : 0;
}

jit_store_emitter::jit_store_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     ov::element::Type src_prc,
                                     ov::element::Type dst_prc,
                                     size_t store_num,
                                     size_t byte_offset,
                                     arithmetic_mode mode,
                                     ov::element::Type exec_prc,
                                     emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type),
      store_num_(store_num),
      byte_offset_(byte_offset),
      src_prc_(src_prc),
      dst_prc_(dst_prc),
      mode_(mode) {
    jit_conversion::validate_convert_precision(src_prc_, dst_prc_);
}

size_t jit_store_emitter::get_inputs_num() const {
    return 1;
}

void jit_store_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::gv) {
        emit_isa<ov::intel_cpu::riscv64::gv>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported isa.");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_store_emitter::emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    auto src = Xbyak_riscv::VReg(in_idxs[0]);
    auto dst = Xbyak_riscv::Reg(out_idxs[0]);
    auto data = src;

    if (needs_conversion(src_prc_, dst_prc_)) {
        OV_CPU_JIT_EMITTER_ASSERT(!aux_vec_idxs.empty(), "Store conversion requires an auxiliary vector register");
        OV_CPU_JIT_EMITTER_ASSERT(!aux_gpr_idxs.empty(), "Store conversion requires an auxiliary GPR");
        data = Xbyak_riscv::VReg(aux_vec_idxs.front());
        set_vector_length(h, store_num_, jit_conversion::byte_size_to_sew(src_prc_.size()), aux_gpr_idxs);
        const auto avl = Xbyak_riscv::Reg(aux_gpr_idxs.front());
        jit_conversion::emit_convert_process(h, src, data, src_prc_, dst_prc_, mode_, avl);
    }

    const auto byte_size = dst_prc_.size();
    set_vector_length(h, store_num_, jit_conversion::byte_size_to_sew(byte_size), aux_gpr_idxs);

    if (byte_offset_ == 0) {
        if (byte_size == 1) {
            h->vse8_v(data, dst);
        } else if (byte_size == 2) {
            h->vse16_v(data, dst);
        } else {
            h->vse32_v(data, dst);
        }
    } else {
        OV_CPU_JIT_EMITTER_ASSERT(!aux_gpr_idxs.empty(), "Static byte offset requires an auxiliary GPR");
        auto tmp_gpr = Xbyak_riscv::Reg(aux_gpr_idxs.front());
        h->uni_li(tmp_gpr, byte_offset_);
        h->add(tmp_gpr, dst, tmp_gpr);
        if (byte_size == 1) {
            h->vse8_v(data, tmp_gpr);
        } else if (byte_size == 2) {
            h->vse16_v(data, tmp_gpr);
        } else {
            h->vse32_v(data, tmp_gpr);
        }
    }
}

size_t jit_store_emitter::aux_gprs_count() const {
    return (store_num_ > 31 || byte_offset_ != 0 || needs_conversion(src_prc_, dst_prc_)) ? 1 : 0;
}

size_t jit_store_emitter::aux_vecs_count() const {
    return needs_conversion(src_prc_, dst_prc_) ? 1 : 0;
}

}  // namespace ov::intel_cpu::riscv64
