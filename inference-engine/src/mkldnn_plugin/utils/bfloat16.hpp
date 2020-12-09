// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include "utils.hpp"
#include "nodes/common/emitter.h"

/**
 * The bfloat16_t class can be used as an arithmetic type. All arithmetic operations goes through conversion to the float data type.
 */


#define BFLOAT16_ROUND_MODE_TO_NEAREST_EVEN

namespace MKLDNNPlugin {
class bfloat16_t {
public:
    constexpr bfloat16_t()
        : m_value{0}
    {
    }
    bfloat16_t(float value) noexcept
            : m_value{
#if defined BFLOAT16_ROUND_MODE_TO_NEAREST
        round_to_nearest(value)
#elif defined BFLOAT16_ROUND_MODE_TO_NEAREST_EVEN
        round_to_nearest_even(value)
#elif defined BFLOAT16_ROUND_MODE_TRUNCATE
        truncate(value)
#else
#error                                                                                             \
    "ROUNDING_MODE must be one of BFLOAT16_ROUND_MODE_TO_NEAREST, BFLOAT16_ROUND_MODE_TO_NEAREST_EVEN, or BFLOAT16_ROUND_MODE_TRUNCATE"
#endif
    }
    {
    }

    operator float() const {
        return F32{uint32_t(m_value) << 16}.vfloat;
    }
    static constexpr bfloat16_t from_bits(uint16_t bits) { return bfloat16_t(bits, true); }
    uint16_t to_bits() const { return m_value; }

    static inline uint16_t round_to_nearest_even(float x) {
        return static_cast<uint16_t>((F32(x).vint + ((F32(x).vint & 0x00010000) >> 1)) >> 16);
    }

    static inline uint16_t round_to_nearest(float x) {
        return static_cast<uint16_t>((F32(x).vint + 0x8000) >> 16);
    }

    static inline uint16_t truncate(float x) { return static_cast<uint16_t>((F32(x).vint) >> 16); }

private:
    constexpr bfloat16_t(uint16_t x, bool)
            : m_value{x}
    {
    }
    union alignas(16) F32 {
        F32(float val)
                : vfloat{val} {
        }

        F32(uint32_t val)
                : vint{val} {
        }
        float vfloat;
        uint32_t vint;
    };
    uint16_t m_value;
};


class jit_emu_vcvtneps2bf16 : public jit_emitter {
public:
    jit_emu_vcvtneps2bf16(mkldnn::impl::cpu::jit_generator* host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::BF16) : jit_emitter(host, host_isa, node, exec_prc) {
        prepare_table();
    };

    size_t get_inputs_num() { return 1; };

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs,
        const std::vector<size_t>& pool_vec_idxs, const std::vector<size_t>& pool_gpr_idxs) {
        if (host_isa_ == mkldnn::impl::cpu::cpu_isa_t::avx512_common) {
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

    void register_table_entries() {
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

    size_t aux_vecs_count() const { return 2; }
};
} // namespace MKLDNNPlugin

/**
 * std::numeric_limits overloaded for better compatibility with template metaprogramming.
 * For example, to make the following template work:
 *  template <typename T>
 *  void someFunction() {
 *      ...
 *      T maxValue = std::numeric_limits<T>::max();
 *      ...
 *  }
 */

namespace std {
template <>
class numeric_limits<MKLDNNPlugin::bfloat16_t> {
public:
    static constexpr bool is_specialized = true;
    static constexpr MKLDNNPlugin::bfloat16_t min() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x007F);
    }
    static constexpr MKLDNNPlugin::bfloat16_t max() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x7F7F);
    }
    static constexpr MKLDNNPlugin::bfloat16_t lowest() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0xFF7F);
    }
    static constexpr int digits = 7;
    static constexpr int digits10 = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr MKLDNNPlugin::bfloat16_t epsilon() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x3C00);
    }
    static constexpr MKLDNNPlugin::bfloat16_t round_error() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x3F00);
    }
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr MKLDNNPlugin::bfloat16_t infinity() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x7F80);
    }
    static constexpr MKLDNNPlugin::bfloat16_t quiet_NaN() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x7FC0);
    }
    static constexpr MKLDNNPlugin::bfloat16_t signaling_NaN() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0x7FC0);
    }
    static constexpr MKLDNNPlugin::bfloat16_t denorm_min() noexcept {
        return MKLDNNPlugin::bfloat16_t::from_bits(0);
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};
} // namespace std

