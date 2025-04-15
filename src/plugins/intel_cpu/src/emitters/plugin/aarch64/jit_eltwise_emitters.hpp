// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_abs_emitter : public jit_emitter {
public:
    jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_clamp_emitter : public jit_emitter {
public:
    jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const float min,
                      const float max,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    float min;
    float max;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const ov::element::Type exec_prc = ov::element::f32);

    jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_equal_emitter : public jit_emitter {
public:
    jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_not_equal_emitter : public jit_emitter {
public:
    jit_not_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const ov::element::Type exec_prc = ov::element::f32);

    jit_not_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_exp_emitter : public jit_emitter {
public:
    jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_elu_emitter : public jit_emitter {
public:
    jit_elu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const float alpha,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_elu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter;
    float alpha;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_floor_emitter : public jit_emitter {
public:
    jit_floor_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_floor_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};
class jit_floor_mod_emitter : public jit_emitter {
public:
    jit_floor_mod_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const ov::element::Type exec_prc = ov::element::f32);

    jit_floor_mod_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};
class jit_ceiling_emitter : public jit_emitter {
public:
    // Constructor with explicit precision
    jit_ceiling_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const ov::element::Type exec_prc = ov::element::f32);

    // Constructor from node
    jit_ceiling_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node);

    // Get number of inputs
    size_t get_inputs_count() const override;

    // Get supported precisions
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    // Implementation of JIT code emission
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    // ISA-specific implementation
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_negative_emitter : public jit_emitter {
public:
    jit_negative_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);

    jit_negative_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_gelu_erf_emitter : public jit_emitter {
public:
    jit_gelu_erf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);

    jit_gelu_erf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_tanh_emitter;

class jit_gelu_tanh_emitter : public jit_emitter {
public:
    jit_gelu_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const ov::element::Type exec_prc = ov::element::f32);

    jit_gelu_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_tanh_emitter> tanh_emitter;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_greater_emitter : public jit_emitter {
public:
    jit_greater_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const ov::element::Type exec_prc = ov::element::f32);

    jit_greater_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_greater_equal_emitter : public jit_emitter {
public:
    jit_greater_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                              dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                              const ov::element::Type exec_prc = ov::element::f32);

    jit_greater_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                              dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                              const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_hswish_emitter : public jit_emitter {
public:
    jit_hswish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const ov::element::Type exec_prc = ov::element::f32);

    jit_hswish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_is_finite_emitter : public jit_emitter {
public:
    jit_is_finite_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const ov::element::Type exec_prc = ov::element::f32);

    jit_is_finite_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_is_nan_emitter : public jit_emitter {
public:
    jit_is_nan_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const ov::element::Type exec_prc = ov::element::f32);

    jit_is_nan_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_maximum_emitter : public jit_emitter {
public:
    jit_maximum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const ov::element::Type exec_prc = ov::element::f32);

    jit_maximum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_minimum_emitter : public jit_emitter {
public:
    jit_minimum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const ov::element::Type exec_prc = ov::element::f32);

    jit_minimum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_mish_emitter : public jit_emitter {
public:
    jit_mish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const ov::element::Type exec_prc = ov::element::f32);

    jit_mish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_is_inf_emitter : public jit_emitter {
public:
    jit_is_inf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node);

    jit_is_inf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const bool detect_negative,
                       const bool detect_positive,
                       const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;

    bool detect_negative;
    bool detect_positive;
};

class jit_less_emitter : public jit_emitter {
public:
    jit_less_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const ov::element::Type exec_prc = ov::element::f32);

    jit_less_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_less_equal_emitter : public jit_emitter {
public:
    jit_less_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                           const ov::element::Type exec_prc = ov::element::f32);

    jit_less_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                           const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_logical_and_emitter : public jit_emitter {
public:
    jit_logical_and_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const ov::element::Type exec_prc = ov::element::f32);

    jit_logical_and_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_logical_or_emitter : public jit_emitter {
public:
    jit_logical_or_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                           const ov::element::Type exec_prc = ov::element::f32);

    jit_logical_or_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                           const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_logical_not_emitter : public jit_emitter {
public:
    jit_logical_not_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const ov::element::Type exec_prc = ov::element::f32);

    jit_logical_not_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_logical_xor_emitter : public jit_emitter {
public:
    jit_logical_xor_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const ov::element::Type exec_prc = ov::element::f32);

    jit_logical_xor_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_mod_emitter : public jit_emitter {
public:
    jit_mod_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_mod_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
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

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);

    jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
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

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    float power;
    float scale;
    float shift;
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_power_dynamic_emitter : public jit_emitter {
public:
    jit_power_dynamic_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                              dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                              const ov::element::Type exec_prc = ov::element::f32);

    jit_power_dynamic_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                              dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                              const std::shared_ptr<ov::Node>& node,
                              const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_relu_emitter : public jit_emitter {
public:
    jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const float alpha,
                     const ov::element::Type exec_prc = ov::element::f32);

    jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    bool is_relu() const;

private:
    float alpha;
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_round_half_away_from_zero_emitter : public jit_emitter {
public:
    jit_round_half_away_from_zero_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                          const ov::element::Type exec_prc = ov::element::f32);

    jit_round_half_away_from_zero_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                          const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_round_half_to_even_emitter : public jit_emitter {
public:
    jit_round_half_to_even_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc = ov::element::f32);

    jit_round_half_to_even_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_select_emitter : public jit_emitter {
public:
    jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const ov::element::Type exec_prc = ov::element::f32);

    jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_sigmoid_emitter : public jit_emitter {
public:
    jit_sigmoid_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const ov::element::Type exec_prc = ov::element::f32);

    jit_sigmoid_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_softplus_emitter : public jit_emitter {
public:
    jit_softplus_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);

    jit_softplus_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_soft_sign_emitter : public jit_emitter {
public:
    jit_soft_sign_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const ov::element::Type exec_prc = ov::element::f32);

    jit_soft_sign_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                          dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                          const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_sqrt_emitter : public jit_emitter {
public:
    jit_sqrt_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const ov::element::Type exec_prc = ov::element::f32);

    jit_sqrt_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_squared_difference_emitter : public jit_emitter {
public:
    jit_squared_difference_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc = ov::element::f32);

    jit_squared_difference_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);

    jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_swish_emitter : public jit_emitter {
public:
    jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const float beta,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_sigmoid_emitter> sigmoid_emitter;

    float beta;
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_tanh_emitter : public jit_emitter {
public:
    jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     ov::element::Type exec_prc = ov::element::f32);

    jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_sigmoid_emitter> sigmoid_emitter;

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

}  // namespace ov::intel_cpu::aarch64
