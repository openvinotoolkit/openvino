// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {

class jit_uni_vcvtneps2bf16 : public jit_emitter {
public:
    jit_uni_vcvtneps2bf16(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
        ov::element::Type exec_prc = ov::element::bf16) : jit_emitter(host, host_isa, exec_prc) {
        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
            prepare_table();
    }

    size_t get_inputs_num() const override { return 1; }

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
            emit_isa<dnnl::impl::cpu::x64::avx512_core>(in_vec_idxs, out_vec_idxs);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in_vec_idxs, out_vec_idxs);
        } else if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in_vec_idxs, out_vec_idxs);
        } else {
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
        using namespace Xbyak;
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

        Vmm in = Vmm(in_vec_idxs[0]);

        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
            Ymm out = Ymm(out_vec_idxs[0]);
            h->vcvtneps2bf16(out, in);
        } else if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
            Zmm aux = Zmm(aux_vec_idxs[0]);
            Zmm aux1 = Zmm(aux_vec_idxs[1]);
            Ymm out = Ymm(out_vec_idxs[0]);

            h->uni_vpsrld(aux, in, 16);
            h->vpandd(aux, aux, table_val("one"));
            h->uni_vmovups(aux1, table_val("even"));
            h->uni_vpaddd(aux, aux1, aux);
            h->uni_vpaddd(aux, in, aux);
            h->vfixupimmps(aux, in, table_val("selector"), 0);
            h->vpsrad(aux, aux, 16);
            h->vpmovdw(out, aux);
        } else {  // round_to_nearest_even emulation
            Vmm aux = Vmm(aux_vec_idxs[0]);
            Xmm out = Xmm(out_vec_idxs[0]);

            if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx2) {
                h->uni_vandps(aux, in, table_val("rounding"));
            } else {
                h->uni_vmovups(aux, in);
                h->uni_vandps(aux, aux, table_val("rounding"));
            }

            h->uni_vpsrld(aux, aux, 1);
            h->uni_vpaddd(aux, aux, in);
            h->uni_vpsrld(aux, aux, 16);

            // dword to word using truncation
            h->uni_vandps(aux, aux, table_val("mask_truncation_word"));
            h->uni_vpackusdw(aux, aux, aux);

            if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx2) {
                h->vpermq(Ymm(aux.getIdx()), Ymm(aux.getIdx()), 0xD8); //11 01 10 00
                h->vextracti128(out, Ymm(aux.getIdx()), 0);
            } else {
                h->uni_vmovups(out, aux);
            }
        }
    }

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
        push_arg_entry_of("rounding", 0x00010000, true);
        push_arg_entry_of("selector", selector_int32, true);
        push_arg_entry_of("mask_truncation_word", 0x0000ffff, true);
    }

    size_t aux_vecs_count() const override {
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
            return 0;
        return host_isa_ == dnnl::impl::cpu::x64::avx512_core ? 2 : 1;
    }
};

}   // namespace intel_cpu
}   // namespace ov
