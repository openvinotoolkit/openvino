// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_abs_emitter : public jit_emitter {
public:
    jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

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

class jit_clamp_emitter : public jit_emitter {
public:
    jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const float min,
                      const float max,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    float min;
    float max;

    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_equal_emitter : public jit_emitter {
public:
    jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const ov::element::Type exec_prc = ov::element::f32);

    jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
            const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

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

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
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

class jit_select_emitter : public jit_emitter {
public:
    jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const ov::element::Type exec_prc = ov::element::f32);

    jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
            const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
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

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter;

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

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_sigmoid_emitter> sigmoid_emitter;

    float beta;
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_tanh_emitter : public jit_emitter {
public:
    jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     ov::element::Type exec_prc = ov::element::f32);

    jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    size_t get_aux_vecs_count() const override;

    size_t get_aux_gprs_count() const override;

    void register_table_entries() override;

    void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    std::unique_ptr<jit_sigmoid_emitter> sigmoid_emitter;

    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
