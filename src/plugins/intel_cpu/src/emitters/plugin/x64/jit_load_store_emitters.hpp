// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include "jit_bf16_emitters.hpp"

namespace ov {
namespace intel_cpu {

struct load_emitter_params : public emitter_params {
    load_emitter_params(ov::element::Type src_prc, ov::element::Type dst_prc,
                        int load_num, bool is_fill = false, std::string fill_value = "zero"):
        src_prc_(src_prc), dst_prc_(dst_prc), load_num_(load_num), is_fill_(is_fill), fill_value_(fill_value) {}

    size_t hash() const override;

    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    int load_num_;
    bool is_fill_;
    std::string fill_value_;
};

struct store_emitter_params : public emitter_params {
    store_emitter_params(ov::element::Type src_prc, ov::element::Type dst_prc, int store_num):
        src_prc_(src_prc), dst_prc_(dst_prc), store_num_(store_num) {}

    size_t hash() const override;

    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    int store_num_;
};

// Arithmetic modes for data type conversion in store_emitter
enum arithmetic_mode {
    saturation,
    truncation
};

class jit_load_emitter : public jit_emitter {
public:
    jit_load_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                     ov::element::Type src_prc, ov::element::Type dst_prc, int load_num,
                     ov::element::Type exec_prc = ov::element::f32,
                     bool is_fill = false, std::string fill_value = "zero",
                     emitter_in_out_map in_out_type = emitter_in_out_map::gpr_to_vec);
    /**
    * load_num values with src_prc precision are loaded from ptr[Reg64(in_idxs[0]) + offset_byte] address to Vmm[out_idxs[0]] as dst_prc, where offset_byte is in_idxs[1]
    * is_fill: when load_num can not fully fit in vector register, whether fill_value should be filled as default values.
    * fill_value: when load_num can not fully fit in vector register, what values should be filled as default values.
    *   currently support "zero", "int_one", "float_one", "int32_min", "float_min", "int32_max" and "float_max".
    * supported src_prc and dst_prc pairs are as below(x indicate for support):
    *       FP32  I32   I16   U16   I8    U8    BF16  --> src_prc
    * FP32   x     x     x     x     x    x     x
    * I32    x     x     x     x     x    x     x
    * I16                x
    * U16                      x
    * I8                             x
    * U8                                  x
    * BF16                                      x
    *  |
    * \|/
    * dst_prc
    */

    // offset in load emitter is the offset of src gpr register, should be parsed from in_idxs.
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;

    size_t get_inputs_num() const override;

private:
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const Xbyak::Reg64 &reg_src,  const int out_vec_idx, const int offset) const;

    template <typename Vmm>
    void load_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, int load_size) const;

    template <typename Vmm>
    void load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_signed, int load_size) const;

    template <typename Vmm>
    void load_words_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, ov::element::Type prc, int load_size) const;

    template <typename Vmm>
    void fill_with_default(const Vmm &vmm, std::string fill_value, const int &load_num) const;

    void register_table_entries() override;

    size_t aux_gprs_count() const override;

    std::string name_;
    int v_len_elt_;  // 4/8/16
    int load_num_;
    int load_size_;
    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    bool is_fill_;
    std::string fill_value_;
};

class jit_store_emitter : public jit_emitter {
public:
    jit_store_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      ov::element::Type src_prc, ov::element::Type dst_prc, int store_num,
                      arithmetic_mode mode = arithmetic_mode::saturation,
                      ov::element::Type exec_prc = ov::element::f32,
                      emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_gpr);

    /**
    * store_num values with src_prc in Vmm[in_vec_idx] is stored to ptr[reg_dst + offset_byte] address as dst_prc data, where offset_byte is in_idxs[1]
    * supported src_prc and dst_prc pairs are as below(x indicate for support):
    *       FP32  I32   I16   U16   I8    U8    BF16  --> src_prc
    * FP32   x     x
    * I32    x     x
    * I16    x     x     x
    * U16    x     x           x
    * I8     x     x                 x
    * U8     x     x                       x
    * BF16   x*    x*                             x
    * \|/
    * dst_prc
    * note: FP32/I32-->BF16(x*) is supported only on at least avx512-core plateform
    */

    // offset in store emitter is the offset of dst gpr register, should be parsed from out_idxs.
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;

    size_t get_inputs_num() const override;

    void emit_data() const override;

    std::shared_ptr<jit_uni_vcvtneps2bf16> get_uni_vcvtneps2bf16() const {
        return uni_vcvtneps2bf16_;
    }

private:
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const int in_vec_idx,  const Xbyak::Reg64 &reg_dst, const int offset) const;

    template <typename Vmm>
    void store_bytes(const Xbyak::Reg64 &reg, int offset, int store_size) const;

    template <typename Vmm>
    void store_dword_to_byte_extension(const Xbyak::Reg64 &reg, int offset, bool is_signed, int store_size) const;

    template <typename Vmm>
    void store_dword_to_word_extension(const Xbyak::Reg64 &reg, int offset, ov::element::Type precision, int store_size) const;

    void register_table_entries() override;

    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;

    inline bool is_saturation() const;
    inline bool is_truncation_emulation() const;

    std::string name_;
    int v_len_elt_;  // 4/8/16
    int store_num_;
    int store_size_;
    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    arithmetic_mode mode_ = arithmetic_mode::saturation;
    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16_;
    // below members are to keep original vector values
    // data_idx is active vec to store.
    // For instructions that change src vec value, use aux_vec(aux_src_idx) as dst vec to keep src vec intact,
    // then proceed with new vec(data_idx) for continued procedure. Guarantee original vec is not needed any more.
    // once updated, set data_reg_updated to true, and no more update needed.
    mutable bool data_reg_updated = false;
    mutable int data_idx = 0;
    mutable int aux_src_idx = 0;
};

}   // namespace intel_cpu
}   // namespace ov
