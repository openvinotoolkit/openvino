// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace MKLDNNPlugin {

class jit_emu_vcvtneps2bf16 : public jit_emitter {
public:
    jit_emu_vcvtneps2bf16(mkldnn::impl::cpu::x64::jit_generator* host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::BF16) : jit_emitter(host, host_isa, node, exec_prc) {
        prepare_table();
    }

    size_t get_inputs_num() const override { return 1; }

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs,
        const std::vector<size_t>& pool_vec_idxs, const std::vector<size_t>& pool_gpr_idxs,
        const emitter_context *emit_context) const override {
        if (host_isa_ == mkldnn::impl::cpu::x64::cpu_isa_t::avx512_common) {
            Xbyak::Zmm in = Xbyak::Zmm(in_vec_idxs[0]);
            Xbyak::Ymm out = Xbyak::Ymm(out_vec_idxs[0]);
            Xbyak::Zmm aux = Xbyak::Zmm(aux_vec_idxs[0]);
            Xbyak::Zmm aux1 = Xbyak::Zmm(aux_vec_idxs[1]);

            h->uni_vpsrld(aux, in, 16);
            h->vpandd(aux, aux, table_val("one"));
            h->uni_vmovups(aux1, table_val("even"));
            h->uni_vpaddd(aux, aux1, aux);
            h->uni_vpaddd(aux, in, aux);
            h->vfixupimmps(aux, in, table_val("selector"), 0);
            h->vpsrad(aux, aux, 16);
            h->vpmovdw(out, aux);
        } else {
            assert(!"unsupported isa");
        }
    };


    inline int encode_fixup_selector(int input, int output) {
        return ((output) << (4 * (input)));
    }

    void register_table_entries() override {
        enum {
            fixup_input_code_qnan_ = 0,
            fixup_input_code_snan_ = 1,
            fixup_input_code_ninf_ = 4,
            fixup_input_code_pinf_ = 5,
            fixup_output_code_copy_input_ = 1,
            fixup_output_code_qnan_input_ = 2,
        };
        const int selector_int32 =
            /* qnan input to qnan output (presenrving input bits 0..21) */
            encode_fixup_selector(fixup_input_code_snan_, fixup_output_code_qnan_input_) |
            /* snan input to qnan output (presenrving input bits 0..21) */
            encode_fixup_selector(fixup_input_code_qnan_, fixup_output_code_qnan_input_) |
            /* neg inf input copied to output */
            encode_fixup_selector(fixup_input_code_ninf_, fixup_output_code_copy_input_) |
            /* pos inf input copied to output */
            encode_fixup_selector(fixup_input_code_pinf_, fixup_output_code_copy_input_);
        push_arg_entry_of("one", 0x00000001, true);
        push_arg_entry_of("even", 0x00007fff, true);
        push_arg_entry_of("selector", selector_int32, true);
    }

    size_t aux_vecs_count() const override { return 2; }
};

} // namespace MKLDNNPlugin