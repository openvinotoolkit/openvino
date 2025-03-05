// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "emitters/utils.hpp"
#include "snippets/generator.hpp"
#include "snippets/snippets_isa.hpp"

#include <set>

namespace ov::intel_cpu::riscv64 {

enum emitter_in_out_map {
    vec_to_vec,
    vec_to_gpr,
    gpr_to_vec,
    gpr_to_gpr,
};

class jit_emitter : public ov::snippets::Emitter {
public:
    jit_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                ov::element::Type exec_prc = ov::element::f32,
                emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec);

    // We have to define second "emit_code" to pass FP registers because
    // the base class method doesn't support them for code emission
    virtual void emit_code(const std::vector<size_t>& in_idxs,
                           const std::vector<size_t>& out_idxs,
                           const std::vector<size_t>& pool_vec_idxs,
                           const std::vector<size_t>& pool_gpr_idxs,
                           const std::vector<size_t>& pool_fp_gpr_idxs) const {
        emit_code_impl(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs, pool_fp_gpr_idxs);
    }

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

    // TODO: RV64 supports vector multiplier.
    // However, currently not all JIT emitter support LMUL > 1:
    //   - if aux_vec registers are needed - preamble/postamble support only m1.
    //     These JIT emitters should known exact value of LMUL to preserve vec regs with correct idxs.
    //   - etc
    virtual bool is_lmul_supported() const {
        return aux_vecs_count() == 0;
    }

protected:
    size_t get_max_gpr_count() const { return 32; }
    size_t get_max_fp_gpr_count() const { return 32; }
    size_t get_max_vecs_count() const { return 32; }

    size_t get_gpr_length() const;
    size_t get_fp_gpr_length() const;
    size_t get_vec_length() const;

    Xbyak_riscv::VReg mask_vreg() const { return Xbyak_riscv::v0; }

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override {
        emit_code_impl(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs, {});
    }

    virtual void emit_code_impl(const std::vector<size_t>& in_idxs,
                                const std::vector<size_t>& out_idxs,
                                const std::vector<size_t>& pool_vec_idxs,
                                const std::vector<size_t>& pool_gpr_idxs,
                                const std::vector<size_t>& pool_fp_gpr_idxs) const;

    virtual void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const = 0;

    virtual void emitter_preamble(const std::vector<size_t>& in_idxs,
                                  const std::vector<size_t>& out_idxs,
                                  const std::vector<size_t>& pool_vec_idxs,
                                  const std::vector<size_t>& pool_gpr_idxs,
                                  const std::vector<size_t>& pool_fp_gpr_idxs) const;
    virtual void emitter_postamble() const;

    void store_context(const std::vector<size_t>& gpr_regs,
                       const std::vector<size_t>& fp_gpr_regs,
                       const std::vector<size_t>& vec_regs) const;
    void restore_context(const std::vector<size_t>& gpr_regs,
                         const std::vector<size_t>& fp_gpr_regs,
                         const std::vector<size_t>& vec_regs) const;

    // Save all caller-saved registers (gp, fp-gp, vec) exclude passed arguments
    // These helpers might be called for save binary call
    void call_preamble(const std::vector<size_t>& exclude_gpr_regs,
                       const std::vector<size_t>& exclude_fp_gpr_regs,
                       const std::vector<size_t>& exclude_vec_regs) const;
    void call_postamble(const std::vector<size_t>& exclude_gpr_regs,
                        const std::vector<size_t>& exclude_fp_gpr_regs,
                        const std::vector<size_t>& exclude_vec_regs) const;

    virtual void validate_arguments(const std::vector<size_t>&, const std::vector<size_t>&) const {}

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t;  // offsets are in bytes wrt p_table

    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
    };

    using table_t = std::multimap<std::string, table_entry_val_t>;
    using mapped_table_t = std::multimap<std::string, mapped_table_entry_t>;

    void push_arg_entry_of(const std::string& key, const table_entry_val_t val) {
        mapped_table_entry_t te{0, val};
        entry_map_.insert(std::make_pair(key, te));
    }

    void push_entries_of(const table_t& t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second;  // copy values from table
            push_arg_entry_of(key, te);
        }
    }

    virtual void prepare_table();
    virtual void register_table_entries() {}

    void load_table_addr() const {
        const auto address = reinterpret_cast<uintptr_t>(l_table->getAddress());
        OPENVINO_ASSERT(address != 0, "Address of data section is missed!");
        h->uni_li(p_table, address);
    }

    inline void load_table_val(const std::string& key, const Xbyak_riscv::FReg& freg, size_t key_off_val_shift = 0) const {
        auto off = table_off(key, key_off_val_shift);
        h->flw(freg, p_table, off);
    }

    inline void load_table_val(const std::string& key, const Xbyak_riscv::Reg& reg, size_t key_off_val_shift = 0) const {
        auto off = table_off(key, key_off_val_shift);
        h->lw(reg, p_table, off);
    }

    // Load scalar to vector with broadcast
    inline void load_table_val(const std::string& key, const Xbyak_riscv::VReg& vreg, const Xbyak_riscv::Reg& tmp, size_t key_off_val_shift = 0) const {
        auto off = table_off(key, key_off_val_shift);
        h->lw(tmp, p_table, off);
        h->vmv_v_x(vreg, tmp);
    }

    ov::intel_cpu::riscv64::jit_generator* h;
    ov::intel_cpu::riscv64::cpu_isa_t host_isa_;
    ov::element::Type exec_prc_;

    mutable std::shared_ptr<Xbyak_riscv::Label> l_table;
    mutable Xbyak_riscv::Reg p_table;
    mutable std::vector<size_t> aux_vec_idxs;
    mutable std::vector<size_t> aux_gpr_idxs;
    mutable std::vector<size_t> aux_fp_gpr_idxs;

    emitter_in_out_map in_out_type_;
    mapped_table_t entry_map_;

private:
    mutable std::vector<size_t> preserved_vec_idxs;
    mutable std::vector<size_t> preserved_gpr_idxs;
    mutable std::vector<size_t> preserved_fp_gpr_idxs;

    // In the standard RISC-V calling convention, the stack pointer is always kept 16-byte aligned
    const size_t sp_aligment = 16;
    // integer gpr byte size
    const size_t xlen = Xbyak_riscv::CPU().getXlen() / 8;
    // fp gpr byte size
    const size_t flen = Xbyak_riscv::CPU().getFlen() / 8;
    // vector register byte size
    const size_t vlen = Xbyak_riscv::CPU().getVlen() / 8;

    size_t table_off(const std::string& key, size_t key_off_val_shift = 0) const {
        const auto it = entry_map_.find(key);  // search an entry for a key
        OV_CPU_JIT_EMITTER_ASSERT(it != entry_map_.end(), "Value has not been found in the table");
        const auto& te = (*it).second;
        const auto scale = sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }
};

}  // ov::intel_cpu::riscv64
