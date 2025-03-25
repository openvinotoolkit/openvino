// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_bf16_emitters.hpp"
#include "jit_emitter.hpp"

namespace ov::intel_cpu {

class jit_convert_emitter : public jit_emitter {
public:
    jit_convert_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                        dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n,
                        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;

protected:
    void emit_data() const override;
    void validate_types() const;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void float2bfloat(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    ov::element::Type input_type;
    ov::element::Type output_type;

    const ov::element::TypeVector supported_types =
        {ov::element::f32, ov::element::i32, ov::element::bf16, ov::element::f16, ov::element::i8, ov::element::u8};

    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16 = nullptr;
};

// This emitter is covered by specification of "Convert" operation. The implementation uses a "warp-around" conversion.
// Example:
//  int32_t -> int8_t
//   129   -> -127
class jit_convert_truncation_emitter : public jit_convert_emitter {
public:
    jit_convert_truncation_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                   dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& n,
                                   ov::element::Type exec_prc = ov::element::f32);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void dword2int8(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    bool is_i8_and_u8_case() const;
    void register_table_entries() override;
};

// This emitter is covered by the common dnnl behavior. The implementation uses a "saturation" conversion.
// Example:
//  int32_t -> int8_t
//   129   -> 127
class jit_convert_saturation_emitter : public jit_convert_emitter {
public:
    jit_convert_saturation_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                   dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& n,
                                   ov::element::Type exec_prc = ov::element::f32);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void dword2int8(const std::vector<size_t>& in_vec_idxs,
                    const std::vector<size_t>& out_vec_idxs,
                    bool is_signed) const;

    size_t aux_vecs_count() const override;
};

}  // namespace ov::intel_cpu
