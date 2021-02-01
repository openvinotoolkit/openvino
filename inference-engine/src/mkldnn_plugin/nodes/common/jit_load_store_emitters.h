// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitter.h"
#include <cpu/x64/jit_generator.hpp>
#include "mkldnn_node.h"
#include "utils/bfloat16.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace InferenceEngine;

namespace MKLDNNPlugin {
struct load_emitter_context : public emitter_context {
    load_emitter_context() : src_prc_(Precision::FP32), dst_prc_(Precision::FP32), load_num_(8),
    offset_byte_(0), is_fill_(false), fill_value_("zero") {}

    load_emitter_context(Precision src_prc, Precision dst_prc, int load_num, bool is_fill = false, std::string fill_value = "zero", int offset_byte = 0):
    src_prc_(src_prc), dst_prc_(dst_prc), load_num_(load_num), is_fill_(is_fill), fill_value_(fill_value), offset_byte_(offset_byte) {}

    int offset_byte_;
    int load_num_;
    Precision src_prc_;
    Precision dst_prc_;
    bool is_fill_;
    std::string fill_value_;
};

struct store_emitter_context : public emitter_context {
    store_emitter_context() : src_prc_(Precision::FP32), dst_prc_(Precision::FP32),
    store_num_(8), offset_byte_(0) {}

    store_emitter_context(Precision src_prc, Precision dst_prc, int store_num, int offset_byte = 0)
    : src_prc_(src_prc), dst_prc_(dst_prc), store_num_(store_num), offset_byte_(offset_byte) {}

    int offset_byte_;
    int store_num_;
    Precision src_prc_;
    Precision dst_prc_;
};

class jit_load_emitter : public jit_emitter {
public:
    jit_load_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32, emitter_in_out_map in_out_type = emitter_in_out_map::gpr_to_vec);
    /**
    * load_num values with src_prc precision are loaded from ptr[Reg64(in_idxs[0]) + offset_byte] address to Vmm[out_idxs[0]] as dst_prc.
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
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) override;

    size_t get_inputs_num() override;

private:
    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const Xbyak::Reg64 &reg_src, int offset_byte, InferenceEngine::Precision src_prc,
        const int out_vec_idx, InferenceEngine::Precision dst_prc, int load_num, bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void load_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, int load_size,
        bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_signed, int load_size,
        bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void load_words_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_bf16, bool is_signed, int load_size,
        bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void fill_with_default(const Vmm &vmm, std::string fill_value, const int &load_num) const;

    void register_table_entries() override;

    size_t aux_gprs_count() const override;

    int v_len_elt;  // 4/8/16
};

class jit_store_emitter : public jit_emitter {
public:
    jit_store_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32, emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_gpr);

    /**
    * store_num values with src_prc in Vmm[in_vec_idx] is stored to ptr[reg_dst + offset_byte] address as dst_prc data.
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
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) override;

    size_t get_inputs_num() override;

    std::shared_ptr<jit_emu_vcvtneps2bf16> get_emu_vcvtneps2bf16() const {
        return emu_vcvtneps2bf16;
    }

private:
    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const int in_vec_idx, InferenceEngine::Precision src_prc,
        const Xbyak::Reg64 &reg_dst, int offset_byte, InferenceEngine::Precision dst_prc, int store_num) const;

    template <typename Vmm>
    void store_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, int store_size) const;

    template <typename Vmm>
    void store_dword_to_byte_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_signed, int store_size) const;

    template <typename Vmm>
    void store_dword_to_word_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int offset, bool is_bf16, bool is_signed, int store_size) const;

    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;

    int v_len_elt;  // 4/8/16
    std::shared_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16;
};

} // namespace MKLDNNPlugin
