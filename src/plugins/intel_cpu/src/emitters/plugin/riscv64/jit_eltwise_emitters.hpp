// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_clamp_emitter : public jit_emitter {
public:
    jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      float min, float max, const ov::element::Type exec_prc = ov::element::f32);
    jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;

    float min {0.f};
    float max {0.f};
};

class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       const ov::element::Type exec_prc = ov::element::f32);
    jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_exp_emitter : public jit_emitter {
public:
    jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const ov::element::Type exec_prc = ov::element::f32);
    jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);
    jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};
class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);
    jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_power_static_emitter : public jit_emitter {
public:
    jit_power_static_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                              float power, float scale, float shift, const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    bool is_lmul_supported() const override;

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;

    inline bool is_sqrt() const { return power == 0.5f || power == -0.5f; }
    inline bool is_int_pow() const { return std::floor(power) == power && power != 0; }
    inline bool is_scale_shift() const { return scale != 1.f || shift != 0.f; }

    float power {1.f};
    float scale {1.f};
    float shift {0.f};
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      const ov::element::Type exec_prc = ov::element::f32);
    jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_relu_emitter : public jit_emitter {
public:
    jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     float alpha, const ov::element::Type exec_prc = ov::element::f32);
    jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;

    float alpha {0.f};
};

class jit_sigmoid_emitter : public jit_emitter {
public:
    jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                        const ov::element::Type exec_prc = ov::element::f32);
    jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void emit_data() const override;

    void register_table_entries() override;

    std::unique_ptr<jit_exp_emitter> jit_exp_emitter_ {nullptr};
};

class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const ov::element::Type exec_prc = ov::element::f32);
    jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

}  // ov::intel_cpu::riscv64

