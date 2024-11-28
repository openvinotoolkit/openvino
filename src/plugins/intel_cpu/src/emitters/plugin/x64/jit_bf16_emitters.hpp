// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {

class jit_uni_vcvtneps2bf16 : public jit_emitter {
public:
    jit_uni_vcvtneps2bf16(dnnl::impl::cpu::x64::jit_generator* host,
                          dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                          ov::element::Type exec_prc = ov::element::bf16,
                          arithmetic_mode mode = arithmetic_mode::saturation)
        : jit_emitter(host, host_isa, exec_prc) {
        mode_ = mode;
        prepare_table();
    }

    size_t get_inputs_num() const override {
        return 1;
    }

private:
    arithmetic_mode mode_ = arithmetic_mode::saturation;
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
            emit_isa<dnnl::impl::cpu::x64::avx512_core>(in_vec_idxs, out_vec_idxs);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in_vec_idxs, out_vec_idxs);
        } else if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in_vec_idxs, out_vec_idxs);
        } else {
            OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
        }
    }

    template <typename Vmm>
    void saturate_input(Vmm& truncated,
                        const Vmm& in,
                        const std::string& bf16_min_key,
                        const std::string& bf16_max_key) const {
        Vmm bf16_max = Vmm(aux_vec_idxs[0]);
        Vmm bf16_min = Vmm(aux_vec_idxs[1]);
        h->uni_vmovups(bf16_max, table_val(bf16_max_key));
        h->uni_vmovups(bf16_min, table_val(bf16_min_key));
        h->uni_vmaxps(truncated, in, bf16_min);
        h->uni_vminps(truncated, truncated, bf16_max);
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
        using namespace Xbyak;
        using Vmm = typename dnnl::impl::utils::
            conditional3<isa == dnnl::impl::cpu::x64::sse41, Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

        Vmm in = Vmm(in_vec_idxs[0]);

        if (mode_ == arithmetic_mode::saturation) {
            Vmm truncated = Vmm(aux_vec_idxs[2]);
            if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
                Ymm out = Ymm(out_vec_idxs[0]);
                saturate_input(truncated, in, "bf16_min", "bf16_max");
                h->vcvtneps2bf16(out, truncated);
            } else if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
                saturate_input(truncated, in, "bf16_min", "bf16_max");

                Zmm aux = Zmm(aux_vec_idxs[3]);
                Zmm aux1 = Zmm(aux_vec_idxs[4]);
                Ymm out = Ymm(out_vec_idxs[0]);

                h->uni_vpsrld(aux, truncated, 16);
                h->vpandd(aux, aux, table_val("one"));
                h->uni_vmovups(aux1, table_val("even"));
                h->uni_vpaddd(aux, aux1, aux);
                h->uni_vpaddd(aux, truncated, aux);
                h->vfixupimmps(aux, truncated, table_val("selector"), 0);
                h->vpsrad(aux, aux, 16);
                h->vpmovdw(out, aux);
            } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2_vnni_2)) {
                saturate_input(truncated, in, "bf16_min", "bf16_max");
                Xmm out = Xmm(out_vec_idxs[0]);
                h->vcvtneps2bf16(out, truncated, PreferredEncoding::VexEncoding);
            } else {
                Xmm out = Xmm(out_vec_idxs[0]);
                Vmm aux = Vmm(aux_vec_idxs[3]);

                saturate_input(truncated, in, "bf16_min", "bf16_max");

                if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx2) {
                    h->uni_vandps(aux, truncated, table_val("rounding"));
                } else {
                    h->uni_vmovups(aux, truncated);
                    h->uni_vandps(aux, aux, table_val("rounding"));
                }

                h->uni_vpsrld(truncated, truncated, 16);
                h->uni_vandps(truncated, truncated, table_val("mask_truncation_word"));
                h->uni_vpackusdw(truncated, truncated, truncated);

                if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx2) {
                    h->vpermq(Ymm(truncated.getIdx()), Ymm(truncated.getIdx()), 0xD8);  // 11 01 10 00
                    h->vextracti128(out, Ymm(truncated.getIdx()), 0);
                } else {
                    h->uni_vmovups(out, truncated);
                }
            }
        } else {
            if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
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
            } else {
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
                    h->vpermq(Ymm(aux.getIdx()), Ymm(aux.getIdx()), 0xD8);  // 11 01 10 00
                    h->vextracti128(out, Ymm(aux.getIdx()), 0);
                } else {
                    h->uni_vmovups(out, aux);
                }
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
        push_arg_entry_of("bf16_max", 0x7F7F0000, true);
        push_arg_entry_of("bf16_min", 0xFF7F0000, true);
    }

    size_t aux_vecs_count() const override {
        if (mode_ == arithmetic_mode::saturation) {
            if (host_isa_ == dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
                return 5;
            } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2_vnni_2) ||
                       dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_bf16)) {
                return 3;
            }
            return 4;
        } else {
            return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 2 : 1;
        }
    }
};

}  // namespace intel_cpu
}  // namespace ov
