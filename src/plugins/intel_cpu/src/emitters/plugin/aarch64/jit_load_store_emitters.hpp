// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

// Arithmetic modes for data type conversion in store_emitter
enum class arithmetic_mode {
    saturation,
    truncation
};

class jit_load_emitter : public jit_emitter {
public:
    jit_load_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     ov::element::Type src_prc, ov::element::Type dst_prc, int load_num, int byte_offset,
                     ov::element::Type exec_prc = ov::element::f32,
                     emitter_in_out_map in_out_type = emitter_in_out_map::gpr_to_vec);

    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
    size_t get_inputs_count() const override { return 1; };

private:
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void load_qbyte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void load_dbyte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void load_byte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    size_t get_aux_gprs_count() const override;

    std::string name_;
    int load_num_;  // the element number to load
    int byte_offset_;
    ov::element::Type prc_;
};

class jit_store_emitter : public jit_emitter {
public:
    jit_store_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      ov::element::Type src_prc, ov::element::Type dst_prc, int store_num, int byte_offset_,
                      arithmetic_mode mode = arithmetic_mode::saturation, ov::element::Type exec_prc = ov::element::f32,
                      emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_gpr);

    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
    size_t get_inputs_count() const override { return 1; }

private:
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void store_qbyte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void store_dbyte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void store_byte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
    size_t get_aux_gprs_count() const override;

    std::string name_;
    int store_num_;  // the element number to store
    int byte_offset_;
    ov::element::Type prc_;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
