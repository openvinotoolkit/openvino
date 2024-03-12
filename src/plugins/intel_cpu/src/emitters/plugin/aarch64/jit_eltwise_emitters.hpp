// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);

    jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);

    jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_power_static_emitter : public jit_emitter {
public:
    jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const float power,
                      const float scale,
                      const float shift,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node,
                      const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    float power;
    float scale;
    float shift;
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_relu_emitter : public jit_emitter {
public:
    jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const ov::element::Type exec_prc = ov::element::f32);

    jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
