// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "emitters/utils.hpp"
#include "snippets/generator.hpp"
#include "snippets/snippets_isa.hpp"

#include <set>

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

    // We have to define two "emit_code" to pass FP registers because
    // the base class method doesn't support them for code emission
    void emit_code(const std::vector<size_t>& in_idxs,
                   const std::vector<size_t>& out_idxs,
                   const std::vector<size_t>& pool_vec_idxs,
                   const std::vector<size_t>& pool_gpr_idxs,
                   const std::vector<size_t>& pool_fp_gpr_idxs) const;
    void emit_code(const std::vector<size_t>& in_idxs,
                   const std::vector<size_t>& out_idxs,
                   const std::vector<size_t>& pool_vec_idxs,
                   const std::vector<size_t>& pool_gpr_idxs) const override;

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
    size_t get_max_gpr_count() const { return 32; }
    size_t get_max_fp_gpr_count() const { return 32; }
    size_t get_max_vecs_count() const { return 32; }

    size_t get_gpr_length() const;
    size_t get_fp_gpr_length() const;
    size_t get_vec_length() const;

    Xbyak_riscv::VReg mask_vreg() const { return Xbyak_riscv::v0; }

    virtual bool need_table() const { return false; }
    virtual const void* get_table() const { return nullptr; };

    void load_table_addr(const void* table_ptr) const {
        h->uni_li(p_table, (size_t)table_ptr);
    }

    virtual void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const = 0;

    virtual void emitter_preamble(const std::vector<size_t>& in_idxs,
                                  const std::vector<size_t>& out_idxs,
                                  const std::vector<size_t>& pool_vec_idxs,
                                  const std::vector<size_t>& pool_gpr_idxs,
                                  const std::vector<size_t>& pool_fp_gpr_idxs) const;
    virtual void emitter_postamble() const;

    void store_context(const std::vector<size_t>& gpr_regs,
                       const std::vector<size_t>& fp_gpr_regs,
                       const std::vector<size_t>& vec_regs,
                       const std::unordered_set<size_t>& ignore_vec_regs = {}) const;
    void restore_context(const std::vector<size_t>& gpr_regs,
                         const std::vector<size_t>& fp_gpr_regs,
                         const std::vector<size_t>& vec_regs,
                         const std::unordered_set<size_t>& ignore_vec_regs = {}) const;

    // Xbyak_riscv64::Address table_val(const std::string& key, size_t key_off_val_shift = 0) const {
    //     auto off = table_off(key, key_off_val_shift);
    //     return h->ptr[p_table + off];
    // }

    virtual void validate_arguments(const std::vector<size_t>&, const std::vector<size_t>&) const {}

    ov::intel_cpu::riscv64::jit_generator* h;
    ov::element::Type exec_prc_;

    mutable Xbyak_riscv::Reg p_table;
    mutable std::shared_ptr<Xbyak_riscv::Label> l_table;
    mutable std::vector<size_t> aux_vec_idxs;
    mutable std::vector<size_t> aux_gpr_idxs;
    mutable std::vector<size_t> aux_fp_gpr_idxs;

    emitter_in_out_map in_out_type_;

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
};

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov
