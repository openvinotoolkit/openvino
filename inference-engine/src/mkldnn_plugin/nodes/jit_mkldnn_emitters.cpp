// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/emitter.h"
#include "jit_mkldnn_emitters.hpp"
#include "mkldnn_eltwise_node.h"
#include "legacy/ie_layers.h"

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace Xbyak;

namespace MKLDNNPlugin {

jit_mkldnn_emitter::jit_mkldnn_emitter(jit_generator *host, cpu_isa_t host_isa, const MKLDNNNode* node, InferenceEngine::Precision exec_prc)
    : jit_emitter(host, host_isa, node, exec_prc) {
    auto& eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode&>(*n);

    auto alg = static_cast<mkldnn_alg_kind_t>(eltwiseNode.getAlgorithm());

    if (host_isa_ == cpu::sse42) {
        eltwise_injector_sse42 = std::make_shared<jit_uni_eltwise_injector_f32<cpu::sse42>>(
                host, alg, eltwiseNode.getAlpha(), eltwiseNode.getBeta());
    } else if (host_isa_ == cpu::avx2) {
        eltwise_injector_avx2 = std::make_shared<jit_uni_eltwise_injector_f32<cpu::avx2>>(
                host, alg, eltwiseNode.getAlpha(), eltwiseNode.getBeta());
    } else if (host_isa_ == cpu::avx512_common) {
        eltwise_injector_avx512_common = std::make_shared<jit_uni_eltwise_injector_f32<cpu::avx512_common>>(
                host, alg, eltwiseNode.getAlpha(), eltwiseNode.getBeta());
    } else {
        assert(!"unsupported isa");
    }
}

size_t jit_mkldnn_emitter::get_inputs_num() { return 1; }

void jit_mkldnn_emitter::emit(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                              const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) {
    if (host_isa_ == cpu::sse42) {
        if (out_vec_idxs[0] != in_vec_idxs[0])
            h->uni_vmovups(Xmm(out_vec_idxs[0]), Xmm(in_vec_idxs[0]));
        eltwise_injector_sse42->compute_vector(out_vec_idxs[0]);
    } else if (host_isa_ == cpu::avx2) {
        if (out_vec_idxs[0] != in_vec_idxs[0])
            h->uni_vmovups(Ymm(out_vec_idxs[0]), Ymm(in_vec_idxs[0]));
        eltwise_injector_avx2->compute_vector(out_vec_idxs[0]);
    } else if (host_isa_ == cpu::avx512_common) {
        if (out_vec_idxs[0] != in_vec_idxs[0])
            h->uni_vmovups(Zmm(out_vec_idxs[0]), Zmm(in_vec_idxs[0]));
        eltwise_injector_avx512_common->compute_vector(out_vec_idxs[0]);
    } else {
        assert(!"unsupported isa");
    }
}

void jit_mkldnn_emitter::emit_table() {
    if (host_isa_ == cpu::sse42) {
        eltwise_injector_sse42->prepare_table();
    } else if (host_isa_ == cpu::avx2) {
        eltwise_injector_avx2->prepare_table();
    } else if (host_isa_ == cpu::avx512_common) {
        eltwise_injector_avx512_common->prepare_table();
    } else {
        assert(!"unsupported isa");
    }
}


} // namespace MKLDNNPlugin
