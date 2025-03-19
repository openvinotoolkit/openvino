// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_dnnl_emitters.hpp"

#include <nodes/eltwise.h>

using namespace dnnl::impl::utils;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu {

std::set<std::vector<element::Type>> jit_dnnl_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

jit_dnnl_emitter::jit_dnnl_emitter(jit_generator* host,
                                   cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      kind(dnnl_eltwise_tanh) {
    set_injector();
}

jit_dnnl_emitter::jit_dnnl_emitter(jit_generator* host,
                                   cpu_isa_t host_isa,
                                   dnnl_alg_kind_t algKind,
                                   float alpha,
                                   float beta,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      kind(algKind),
      alpha(alpha),
      beta(beta) {
    set_injector();
}

void jit_dnnl_emitter::set_injector() {
    if (host_isa_ == cpu::x64::sse41) {
        eltwise_injector_sse42 =
            std::make_shared<jit_uni_eltwise_injector<cpu::x64::sse41>>(h, kind, alpha, beta, 1.f, data_type::f32);
    } else if (host_isa_ == cpu::x64::avx2) {
        eltwise_injector_avx2 =
            std::make_shared<jit_uni_eltwise_injector<cpu::x64::avx2>>(h, kind, alpha, beta, 1.f, data_type::f32);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        eltwise_injector_avx512_core =
            std::make_shared<jit_uni_eltwise_injector<cpu::x64::avx512_core>>(h,
                                                                              kind,
                                                                              alpha,
                                                                              beta,
                                                                              1.f,
                                                                              data_type::f32);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

size_t jit_dnnl_emitter::get_inputs_num() const {
    return 1;
}

void jit_dnnl_emitter::emit_code_impl(const std::vector<size_t>& in_vec_idxs,
                                      const std::vector<size_t>& out_vec_idxs,
                                      const std::vector<size_t>& pool_vec_idxs,
                                      const std::vector<size_t>& pool_gpr_idxs) const {
    if (host_isa_ == cpu::x64::sse41) {
        if (out_vec_idxs[0] != in_vec_idxs[0]) {
            h->uni_vmovups(Xmm(out_vec_idxs[0]), Xmm(in_vec_idxs[0]));
        }
        eltwise_injector_sse42->compute_vector(out_vec_idxs[0]);
    } else if (host_isa_ == cpu::x64::avx2) {
        if (out_vec_idxs[0] != in_vec_idxs[0]) {
            h->uni_vmovups(Ymm(out_vec_idxs[0]), Ymm(in_vec_idxs[0]));
        }
        eltwise_injector_avx2->compute_vector(out_vec_idxs[0]);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        if (out_vec_idxs[0] != in_vec_idxs[0]) {
            h->uni_vmovups(Zmm(out_vec_idxs[0]), Zmm(in_vec_idxs[0]));
        }
        eltwise_injector_avx512_core->compute_vector(out_vec_idxs[0]);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

void jit_dnnl_emitter::emit_data() const {
    if (host_isa_ == cpu::x64::sse41) {
        eltwise_injector_sse42->prepare_table();
    } else if (host_isa_ == cpu::x64::avx2) {
        eltwise_injector_avx2->prepare_table();
    } else if (host_isa_ == cpu::x64::avx512_core) {
        eltwise_injector_avx512_core->prepare_table();
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

jit_dnnl_aux_emitter::jit_dnnl_aux_emitter(jit_generator* host,
                                           cpu_isa_t host_isa,
                                           dnnl_alg_kind_t algKind,
                                           float inpAlpha,
                                           float inpBeta,
                                           ov::element::Type exec_prc)
    : jit_dnnl_emitter(host, host_isa, algKind, inpAlpha, inpBeta, exec_prc) {}

}  // namespace ov::intel_cpu
