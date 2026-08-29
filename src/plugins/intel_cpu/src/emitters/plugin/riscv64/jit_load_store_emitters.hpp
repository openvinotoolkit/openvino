// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <vector>

#include "emitters/plugin/riscv64/jit_conversion_helpers.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_load_emitter : public jit_emitter {
public:
    jit_load_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     ov::element::Type src_prc,
                     ov::element::Type dst_prc,
                     size_t load_num,
                     size_t byte_offset,
                     arithmetic_mode mode = arithmetic_mode::saturation,
                     ov::element::Type exec_prc = ov::element::f32,
                     emitter_in_out_map in_out_type = emitter_in_out_map::gpr_to_vec);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const;

    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;

    size_t load_num_ = 0;
    size_t byte_offset_ = 0;
    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    arithmetic_mode mode_ = arithmetic_mode::saturation;
};

class jit_store_emitter : public jit_emitter {
public:
    jit_store_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      ov::element::Type src_prc,
                      ov::element::Type dst_prc,
                      size_t store_num,
                      size_t byte_offset,
                      arithmetic_mode mode = arithmetic_mode::saturation,
                      ov::element::Type exec_prc = ov::element::f32,
                      emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_gpr);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const;

    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;

    size_t store_num_ = 0;
    size_t byte_offset_ = 0;
    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    arithmetic_mode mode_ = arithmetic_mode::saturation;
};

}  // namespace ov::intel_cpu::riscv64
