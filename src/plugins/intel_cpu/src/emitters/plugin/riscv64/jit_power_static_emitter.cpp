// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_power_static_emitter.hpp"

#include <memory>
#include "common/utils.hpp"
#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

jit_power_static_emitter::jit_power_static_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                                                   const std::shared_ptr<ov::Node>& node)
                                                   : jit_emitter(host, node, get_input_precision(node)) {
    auto powerStaticNode = ov::as_type_ptr<ov::snippets::op::PowerStatic>(node);
    if (powerStaticNode == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to snippets::op::PowerStatic");
    }

    power = powerStaticNode->get_power();
    scale = 1.f;
    shift = 0.f;
}

jit_power_static_emitter::jit_power_static_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                                                   const float power,
                                                   const float scale,
                                                   const float shift,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, exec_prc),
                                                     power(power),
                                                     scale(scale),
                                                     shift(shift) {
}

size_t jit_power_static_emitter::get_inputs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_gprs_count() const { return 1; }

void jit_power_static_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    emit_isa(in_vec_idxs, out_vec_idxs);
}

void jit_power_static_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());
    OV_CPU_JIT_EMITTER_ASSERT((power == -1) || (power == 0) || (power == 1), "unsupported power: " + std::to_string(power));

    VReg src0 = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    VReg aux = VReg(aux_vec_idxs[0]);

    if (power == 0.f) {
        load(dst, 1.f, Reg(aux_gpr_idxs[0]));
        return;
    }

    bool get_from_dst = false;
    const auto src = [&in_vec_idxs, &out_vec_idxs, &get_from_dst]() -> VReg {
        return get_from_dst ? VReg(out_vec_idxs[0]) : VReg(in_vec_idxs[0]);
    };

    if (scale != 1.f) {
        load(aux, scale, Reg(aux_gpr_idxs[0]));
        h->vfmul_vv(dst, src(), aux);
        get_from_dst = true;
    }

    if (shift != 0.f) {
        load(aux, shift, Reg(aux_gpr_idxs[0]));
        h->vfadd_vv(dst, src(), aux);
        get_from_dst = true;
    }

    if (power == -1.f) {
        load(aux, 1.f, Reg(aux_gpr_idxs[0]));
        h->vfdiv_vv(dst, aux, src());
        get_from_dst = true;
    }
}

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
