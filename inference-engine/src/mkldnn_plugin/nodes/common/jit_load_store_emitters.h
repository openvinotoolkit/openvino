// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitter.h"
#include "jit_generator.hpp"
#include "mkldnn_node.h"
#include "utils/bfloat16.hpp"

namespace MKLDNNPlugin {
struct load_emitter_context : public emitter_context {
    load_emitter_context() : offset_byte_(0), load_num_(8), src_prc_(InferenceEngine::Precision::FP32),
    dst_prc_(InferenceEngine::Precision::FP32), is_fill_(false), fill_value_("zero") {}

    int offset_byte_;
    int load_num_;
    InferenceEngine::Precision src_prc_;
    InferenceEngine::Precision dst_prc_;
    bool is_fill_;
    std::string fill_value_;
};

struct store_emitter_context : public emitter_context {
    store_emitter_context() : offset_byte_(0), store_num_(8), src_prc_(InferenceEngine::Precision::FP32),
    dst_prc_(InferenceEngine::Precision::FP32) {}

    int offset_byte_;
    int store_num_;
    InferenceEngine::Precision src_prc_;
    InferenceEngine::Precision dst_prc_;
};

class jit_load_emitter : public jit_emitter {
public:
    jit_load_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    /**
    * load_num values with src_prc precision are loaded from ptr[Reg64(in_idxs[0]) + offset_byte] address to Vmm[out_idxs[0]] as dst_prc.
    * support dst_prc: FP32 or I32 or src_prc
    * support src_prc: FP32 I32 I16 U16 I8 U8 BF16(on avx512_core || avx512_bf16).
    * is_fill: when load_num can not fully fit in vector register, whether fill_value should be filled as default values.
    * fill_value: when load_num can not fully fit in vector register, what values should be filled as default values.
    *   currently support "zero", "int_one", "float_one", "int32_min", "float_min", "int32_max" and "float_max".
    */
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) override;

    size_t get_inputs_num() override;

private:
    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const Xbyak::Reg64 &reg_src, size_t offset_byte, InferenceEngine::Precision src_prc,
        const int out_vec_idx, InferenceEngine::Precision dst_prc, int load_num, bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void load_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size,
        bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, bool is_signed, int load_size,
        bool is_fill = false, std::string fill_value = "zero") const;

    template <typename Vmm>
    void load_words_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, bool is_bf16, bool is_signed, int load_size,
        bool is_fill = false, std::string fill_value = "zero") const;

    void register_table_entries() override {
        push_arg_entry_of("zero", 0x00000000, true);
        push_arg_entry_of("int_one", 0x00000001, true);
        push_arg_entry_of("float_one", 0x3f800000, true);
        push_arg_entry_of("int32_min", 0xcf000000, true);
        push_arg_entry_of("float_min", 0xff7fffff, true);
        push_arg_entry_of("int32_max", 0x4effffff, true);
        push_arg_entry_of("float_max", 0x7f7fffff, true);
    }

    size_t aux_gprs_count() const override;

    int v_len_elt;  // 4/8/16
};

class jit_store_emitter : public jit_emitter {
public:
    jit_store_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    /**
    * store_num values with src_prc in Vmm[in_vec_idx] is stored to ptr[reg_dst + offset_byte] address as dst_prc data.
    * support src_prc to dst_prc pair: FP32 to all, I32 to all, I16 to I16, U16 to U16, I8 to I8, U8 to U8, BF16 to BF16.
    * i.e. src_prc should be dst_prc or FP32 or I32
    */
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) override;

    size_t get_inputs_num() override;

    std::shared_ptr<jit_emu_vcvtneps2bf16> get_emu_vcvtneps2bf16() const {
        return emu_vcvtneps2bf16;
    }

private:
    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const int in_vec_idx, InferenceEngine::Precision src_prc,
        const Xbyak::Reg64 &reg_dst, size_t offset_byte, InferenceEngine::Precision dst_prc, int store_num) const;

    template <typename Vmm>
    void store_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) const;

    template <typename Vmm>
    void store_dword_to_byte_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, bool is_signed, int store_size) const;

    template <typename Vmm>
    void store_dword_to_word_extension(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, bool is_bf16, bool is_signed, int store_size) const;

    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;

    int v_len_elt;  // 4/8/16
    std::shared_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16;
};

} // namespace MKLDNNPlugin
