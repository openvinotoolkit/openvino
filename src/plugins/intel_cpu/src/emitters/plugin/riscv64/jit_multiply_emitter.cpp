// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_multiply_emitter.hpp"

#include <memory>
#include "common/utils.hpp"
#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

jit_multiply_emitter::jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                                 const std::shared_ptr<ov::Node>& node)
                                 : jit_emitter(host, node, get_input_precision(node)) {
}

jit_multiply_emitter::jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                                 const ov::element::Type exec_prc) : jit_emitter(host, exec_prc) {
}

size_t jit_multiply_emitter::get_inputs_count() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    emit_isa(in_vec_idxs, out_vec_idxs);
}

void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    std::cout << "debug: jit_multiply_emitter::emit_isa: " << exec_prc_ << std::endl;
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfmul_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
