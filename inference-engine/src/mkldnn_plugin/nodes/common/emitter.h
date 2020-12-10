// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include "jit_generator.hpp"
#include "mkldnn_node.h"
#include <set>

namespace MKLDNNPlugin {

class jit_emitter {
public:
    jit_emitter(mkldnn::impl::cpu::jit_generator* host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : h(host), host_isa_(host_isa), n(node), exec_prc_(exec_prc) {
        k_mask = Xbyak::Opmask(1); // FIXME: in general case we need preserve k_mask state as well
    }

    virtual void emit(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                      const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {});
    virtual void emit_table();
    virtual size_t get_inputs_num() = 0;
    virtual size_t aux_vecs_count() const;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

protected:
    virtual size_t aux_gprs_count() const;

    size_t get_max_vecs_count() const;
    size_t get_vec_length() const;

    const MKLDNNNode* n;
    mkldnn::impl::cpu::jit_generator* h;
    mkldnn::impl::cpu::cpu_isa_t host_isa_;
    InferenceEngine::Precision exec_prc_;

    Xbyak::Opmask k_mask;

    virtual void prepare_table();
    virtual void register_table_entries() {}

    void load_table_addr() { h->mov(p_table, l_table); }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    Xbyak::Reg64 p_table;
    Xbyak::Label l_table;

    enum {
        _cmp_eq_oq = mkldnn::impl::cpu::jit_generator::_cmp_eq_oq,
        _cmp_neq_uq = mkldnn::impl::cpu::jit_generator::_cmp_neq_uq,
        _cmp_lt_os = mkldnn::impl::cpu::jit_generator::_cmp_lt_os,
        _cmp_le_os = mkldnn::impl::cpu::jit_generator::_cmp_le_os,
        _cmp_ge_os = mkldnn::impl::cpu::jit_generator::_cmp_nlt_us,
        _cmp_gt_os = mkldnn::impl::cpu::jit_generator::_cmp_nle_us,
    };

    virtual void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                           const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) {}

    virtual void emitter_preamble(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &pool_vec_idxs,
                          const std::vector<size_t> &pool_gpr_idxs);
    virtual void emitter_postamble();

    std::vector<size_t> aux_vec_idxs;
    std::vector<size_t> aux_gpr_idxs;

    static constexpr int k_mask_size = 8;

    Xbyak::Address table_val(std::string key, size_t key_off_val_shift = 0) const {
        auto off = table_off(key, key_off_val_shift);
        return h->ptr[p_table + off];
    }

    using table_t = std::multimap<std::string, table_entry_t>;
    using mapped_table_t = std::multimap<std::string, mapped_table_entry_t>;

    mapped_table_t entry_map_;

    void push_arg_entry_of(const std::string key, const table_entry_val_t val, const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    }

    void push_entries_of(const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    }

private:
    std::vector<size_t> preserved_vec_idxs;
    std::vector<size_t> preserved_gpr_idxs;

    void push_vec(const Xbyak::Address &addr, size_t vec_idx) const;
    void pop_vec(size_t vec_idx, const Xbyak::Address &addr) const;

    size_t table_off(std::string& key, size_t key_off_val_shift = 0) const {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }
};

} // namespace MKLDNNPlugin
