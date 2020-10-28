// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitter.h"
#include <vector>

using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl;
using namespace Xbyak;

namespace MKLDNNPlugin {

template <typename T, typename P>
constexpr bool one_of(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}


size_t jit_emitter::get_max_vecs_count() const {
    return one_of(host_isa_, cpu::avx512_common, cpu::avx512_core) ? 32 : 16;
}

size_t jit_emitter::get_vec_length() const {
    return one_of(host_isa_, cpu::avx512_common, cpu::avx512_core) ? 64 :
           one_of(host_isa_, cpu::avx2) ? 32 : 16;
}

void jit_emitter::push_vec(const Xbyak::Address &addr, size_t vec_idx) const {
    if (host_isa_ == cpu::sse42) {
        h->uni_vmovups(addr, Xmm(vec_idx));
    } else if (host_isa_ == cpu::avx2) {
        h->uni_vmovups(addr, Ymm(vec_idx));
    } else {
        h->uni_vmovups(addr, Zmm(vec_idx));
    }
}

void jit_emitter::pop_vec(size_t vec_idx, const Xbyak::Address &addr) const {
    if (host_isa_ == cpu::sse42) {
        h->uni_vmovups(Xmm(vec_idx), addr);
    } else if (host_isa_ == cpu::avx2) {
        h->uni_vmovups(Ymm(vec_idx), addr);
    } else {
        h->uni_vmovups(Zmm(vec_idx), addr);
    }
}

size_t jit_emitter::aux_vecs_count() const {
    return 0;
}

size_t jit_emitter::aux_gprs_count() const {
    // We need one gpr to load table address
    return entry_map_.empty() ? 0 : 1;
}

std::set<InferenceEngine::Precision> jit_emitter::get_supported_precisions() {
    return {InferenceEngine::Precision::FP32};
}

void jit_emitter::emitter_preamble(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &pool_vec_idxs,
                                   const std::vector<size_t> &pool_gpr_idxs) {
    using namespace Xbyak::util;

    for (auto idx : pool_vec_idxs)
        aux_vec_idxs.push_back(idx);

    // For sse42 mask register has to be Xmm(0)
    if (host_isa_ == cpu::sse42 && aux_vecs_count() > 0) {
        size_t idx = 0;
        assert(std::find(in_vec_idxs.begin(), in_vec_idxs.end(), idx) == in_vec_idxs.end());
        if (std::find(aux_vec_idxs.begin(), aux_vec_idxs.end(), idx) == aux_vec_idxs.end()) {
            aux_vec_idxs.push_back(idx);
            preserved_vec_idxs.push_back(idx);
        }

        // moving mask vector at the beginning of aux vectors list to simplify further processing
        for (int i = 0; i < aux_vec_idxs.size(); i++) {
            if (aux_vec_idxs[i] == 0) {
                size_t tmp = aux_vec_idxs[0];
                aux_vec_idxs[0] = aux_vec_idxs[i];
                aux_vec_idxs[i] = tmp;
                break;
            }
        }
    }

    for (size_t idx = 0; idx < get_max_vecs_count(); idx++) {
        if (aux_vec_idxs.size() >= aux_vecs_count()) break;

        if (std::find(in_vec_idxs.begin(), in_vec_idxs.end(), idx) != in_vec_idxs.end()) continue;
        if (std::find(aux_vec_idxs.begin(), aux_vec_idxs.end(), idx) != aux_vec_idxs.end()) continue;
        if (std::find(preserved_vec_idxs.begin(), preserved_vec_idxs.end(), idx) != preserved_vec_idxs.end()) continue;

        aux_vec_idxs.push_back(idx);
        preserved_vec_idxs.push_back(idx);
    }
    assert(aux_vec_idxs.size() >= aux_vecs_count());

    // Same logic but to allocate gprs
    for (auto idx : pool_gpr_idxs)
        aux_gpr_idxs.push_back(idx);

    for (size_t gpr_idx = 0; gpr_idx <= Operand::R15; ++gpr_idx) {
        size_t _idx = Operand::R15 - gpr_idx; // we allocate from the end

        if (aux_gpr_idxs.size() >= aux_gprs_count()) break;
        if (_idx == Operand::RSP) continue;
        if (std::find(aux_gpr_idxs.begin(), aux_gpr_idxs.end(), _idx) != aux_gpr_idxs.end()) continue;
        if (std::find(preserved_gpr_idxs.begin(), preserved_gpr_idxs.end(), _idx) != preserved_gpr_idxs.end()) continue;

        aux_gpr_idxs.push_back(_idx);
        preserved_gpr_idxs.push_back(_idx);
    }
    assert(aux_gpr_idxs.size() == aux_gprs_count());

    if (!entry_map_.empty()) {
        p_table = Reg64(aux_gpr_idxs[0]);
        aux_gpr_idxs.erase(aux_gpr_idxs.begin());
    }

    for (size_t i = 0; i < preserved_gpr_idxs.size(); ++i)
        h->push(Reg64(preserved_gpr_idxs[i]));

    if (preserved_vec_idxs.size())
        h->sub(h->rsp, preserved_vec_idxs.size() * get_vec_length());

    for (size_t i = 0; i < preserved_vec_idxs.size(); ++i) {
        push_vec(h->ptr[h->rsp + i * get_vec_length()], preserved_vec_idxs[i]);
    }

    if (!entry_map_.empty())
        load_table_addr();
}


void jit_emitter::emitter_postamble() {
    using namespace Xbyak::util;

    for (size_t i = 0; i < preserved_vec_idxs.size(); ++i)
        pop_vec(preserved_vec_idxs[i], h->ptr[h->rsp + i * get_vec_length()]);

    if (preserved_vec_idxs.size())
        h->add(h->rsp, preserved_vec_idxs.size() * get_vec_length());

    for (int i = aux_gprs_count() - 1; i >= 0; --i)
        h->pop(Reg64(preserved_gpr_idxs[i]));

    preserved_vec_idxs.clear();
    preserved_gpr_idxs.clear();

    aux_vec_idxs.clear();
    aux_gpr_idxs.clear();
}

void jit_emitter::emit_table() {
    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);
    }
}

void jit_emitter::prepare_table() {
    register_table_entries();

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
    }
}

void jit_emitter::emit(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                       const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) {
    emitter_preamble(in_vec_idxs, pool_vec_idxs, pool_gpr_idxs);

    emit_impl(in_vec_idxs, out_vec_idxs, pool_vec_idxs, pool_gpr_idxs);

    emitter_postamble();
}

} // namespace MKLDNNPlugin
