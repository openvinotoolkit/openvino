// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <set>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "emitters/utils.hpp"
#include "snippets/generator.hpp"
#include "snippets/snippets_isa.hpp"


namespace ov {
namespace intel_cpu {
namespace riscv64 {

enum emitter_in_out_map {
    vec_to_vec,
    vec_to_gpr,
    gpr_to_vec,
    gpr_to_gpr,
};

// structure for storage of emitter parameters to hash in map
struct emitter_params {
    virtual size_t hash() const = 0;
};

class jit_emitter : public ov::snippets::Emitter {
public:
    jit_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                ov::element::Type exec_prc = ov::element::f32,
                emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec);

    void emit_code(const std::vector<size_t>& in_idxs,
                   const std::vector<size_t>& out_idxs,
                   const std::vector<size_t>& pool_vec_idxs = {},
                   const std::vector<size_t>& pool_gpr_idxs = {}) const override;
    void emit_data() const override;

    virtual size_t get_inputs_num() const = 0;
    virtual size_t aux_vecs_count() const;
    virtual size_t aux_gprs_count() const;
    virtual size_t aux_fp_gprs_count() const;
    emitter_in_out_map get_in_out_type() const;

    /**
     * @brief Returns supported precisions.
     * Precisions are ordered, the first bigger bitness precision with the same type will be selected.
     * Empty collection means the emitter supports any input precisions.
     */
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    size_t get_max_vecs_count() const;
    size_t get_gpr_length() const;
    size_t get_fp_gpr_length() const;
    size_t get_vec_length() const;

    virtual void prepare_table();
    virtual void register_table_entries() {}

    void load_table_addr() const {
        //h->sd(p_table, *l_table.get());
    }

    virtual void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const = 0;

    virtual void emitter_preamble(const std::vector<size_t>& in_idxs,
                                  const std::vector<size_t>& out_idxs,
                                  const std::vector<size_t>& pool_vec_idxs,
                                  const std::vector<size_t>& pool_gpr_idxs) const;
    virtual void emitter_postamble() const;

    void store_context(const std::vector<size_t>& gpr_regs,
                       const std::vector<size_t>& vec_regs,
                       const std::unordered_set<size_t>& ignore_vec_regs = {}) const;
    void restore_context(const std::vector<size_t>& gpr_regs,
                         const std::vector<size_t>& vec_regs,
                         const std::unordered_set<size_t>& ignore_vec_regs = {}) const;

    // Xbyak_riscv64::Address table_val(const std::string& key, size_t key_off_val_shift = 0) const {
    //     auto off = table_off(key, key_off_val_shift);
    //     return h->ptr[p_table + off];
    // }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t;  // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool;     // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    using table_t = std::multimap<std::string, table_entry_t>;
    using mapped_table_t = std::multimap<std::string, mapped_table_entry_t>;

    void push_arg_entry_of(const std::string& key, const table_entry_val_t val, const bool broadcast) {
        mapped_table_entry_t te{0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    }

    void push_entries_of(const table_t& t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second;  // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    }

    virtual void validate_arguments(const std::vector<size_t>&, const std::vector<size_t>&) const {}

    ov::intel_cpu::riscv64::jit_generator* h;
    ov::element::Type exec_prc_;

    mutable Xbyak_riscv::Reg p_table;
    mutable std::shared_ptr<Xbyak_riscv::Label> l_table;
    mutable std::vector<size_t> aux_vec_idxs;
    mutable std::vector<size_t> aux_gpr_idxs;

    mapped_table_t entry_map_;
    emitter_in_out_map in_out_type_;

private:
    mutable std::vector<size_t> preserved_vec_idxs;
    mutable std::vector<size_t> preserved_gpr_idxs;

    size_t table_off(const std::string& key, size_t key_off_val_shift = 0) const {
        const auto it = entry_map_.find(key);  // search an entry for a key
        OV_CPU_JIT_EMITTER_ASSERT(it != entry_map_.end(), "Value has not been found in the table");
        const auto& te = (*it).second;
        const auto scale = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }

    // In the standard RISC-V calling convention, the stack pointer is always kept 16-byte aligned
    const size_t sp_aligment = 16;
    // integer gpr byte size
    const size_t xlen = Xbyak_riscv::CPU().getXlen() / 8;
    // fp gpr byte size
    const size_t flen = Xbyak_riscv::CPU().getFlen() / 8;
    // vector register byte size
    const size_t vlen = Xbyak_riscv::CPU().getVlen() / 8;
};

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov
