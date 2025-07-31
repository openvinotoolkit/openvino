// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_exp_emitter;

class jit_abs_emitter : public jit_emitter {
public:
    jit_abs_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_abs_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
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
    jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      float min,
                      float max,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node,
                      ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;

    float min{0.F};
    float max{0.F};
};

class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       ov::element::Type exec_prc = ov::element::f32);
    jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_elu_emitter : public jit_emitter {
public:
    jit_elu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    float alpha,
                    ov::element::Type exec_prc = ov::element::f32);

    jit_elu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node,
                    ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;

    size_t aux_gprs_count() const override;

    size_t aux_vecs_count() const override;

    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    void emit_data() const override;

private:
    float alpha;
    std::unique_ptr<jit_exp_emitter> exp_emitter{nullptr};

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_equal_emitter : public jit_emitter {
public:
    jit_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_erf_emitter : public jit_emitter {
public:
    jit_erf_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);

    jit_erf_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node,
                    ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;

    size_t aux_gprs_count() const override;

    size_t aux_vecs_count() const override;

    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    void emit_data() const override;

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter{nullptr};

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_exp_emitter : public jit_emitter {
public:
    jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node,
                    ov::element::Type exec_prc = ov::element::f32);

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

class jit_floor_emitter : public jit_emitter {
public:
    jit_floor_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc = element::f32);
    jit_floor_emitter(jit_generator_t* host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
    size_t aux_vecs_count() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;
};
class jit_greater_equal_emitter : public jit_emitter {
public:
    jit_greater_equal_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc = element::f32);
    jit_greater_equal_emitter(jit_generator_t* host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_hsigmoid_emitter : public jit_emitter {
public:
    jit_hsigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);
    jit_hsigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node,
                         ov::element::Type exec_prc = ov::element::f32);

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    size_t get_inputs_num() const override;

    size_t aux_fp_gprs_count() const override;

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_hswish_emitter : public jit_emitter {
public:
    jit_hswish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       ov::element::Type exec_prc = ov::element::f32);
    jit_hswish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& node,
                       ov::element::Type exec_prc = ov::element::f32);

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    size_t get_inputs_num() const override;

    size_t aux_gprs_count() const override;

    size_t aux_vecs_count() const override;

    size_t aux_fp_gprs_count() const override;

    void emit_data() const override;

private:
    std::unique_ptr<jit_hsigmoid_emitter> hsigmoid_emitter{nullptr};

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_less_equal_emitter : public jit_emitter {
public:
    jit_less_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                           ov::element::Type exec_prc = ov::element::f32);
    jit_less_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                           const std::shared_ptr<ov::Node>& node);
    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_maximum_emitter : public jit_emitter {
public:
    jit_maximum_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc = element::f32);
    jit_maximum_emitter(jit_generator_t* host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};
class jit_minimum_emitter : public jit_emitter {
public:
    jit_minimum_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc = element::f32);
    jit_minimum_emitter(jit_generator_t* host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_mod_emitter : public jit_emitter {
public:
    jit_mod_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_mod_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                    ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
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

class jit_logical_and_emitter : public jit_emitter {
public:
    jit_logical_and_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                            ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);
    jit_logical_and_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                            ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
    size_t aux_gprs_count() const override;
    size_t aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_logical_not_emitter : public jit_emitter {
public:
    jit_logical_not_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                            ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);
    jit_logical_not_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                            ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_logical_xor_emitter : public jit_emitter {
public:
    jit_logical_xor_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                            ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);
    jit_logical_xor_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                            ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;
    size_t aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_mish_emitter : public jit_emitter {
public:
    jit_mish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     ov::element::Type exec_prc = ov::element::f32);
    jit_mish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node,
                     ov::element::Type exec_prc = ov::element::f32);

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    size_t get_inputs_num() const override;

    size_t aux_gprs_count() const override;

    size_t aux_vecs_count() const override;

    size_t aux_fp_gprs_count() const override;

    void emit_data() const override;

private:
    std::unique_ptr<jit_exp_emitter> exp_emitter{nullptr};

    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                        ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);
    jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                        ov::intel_cpu::riscv64::cpu_isa_t host_isa,
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
    jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);
    jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_negative_emitter : public jit_emitter {
public:
    jit_negative_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc = element::f32);
    jit_negative_emitter(jit_generator_t* host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_not_equal_emitter : public jit_emitter {
public:
    jit_not_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                          ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                          const std::shared_ptr<ov::Node>& node,
                          ov::element::Type exec_prc);
    jit_not_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                          ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                          ov::element::Type exec_prc);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_power_static_emitter : public jit_emitter {
public:
    jit_power_static_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                             ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                             float power,
                             float scale,
                             float shift,
                             ov::element::Type exec_prc = ov::element::f32);

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

    bool is_sqrt() const {
        return power == 0.5F || power == -0.5F;
    }
    bool is_int_pow() const {
        return std::floor(power) == power && power != 0;
    }
    bool is_scale_shift() const {
        return scale != 1.F || shift != 0.F;
    }

    float power{1.F};
    float scale{1.F};
    float shift{0.F};
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node,
                      ov::element::Type exec_prc = ov::element::f32);

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
    jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     float alpha,
                     ov::element::Type exec_prc = ov::element::f32);
    jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node,
                     ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    size_t aux_fp_gprs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;

    void register_table_entries() override;

    float alpha{0.F};
};

class jit_sigmoid_emitter : public jit_emitter {
public:
    jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                        ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);
    jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                        ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node,
                        ov::element::Type exec_prc = ov::element::f32);

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

    std::unique_ptr<jit_exp_emitter> jit_exp_emitter_{nullptr};
};

class jit_sqrt_emitter : public jit_emitter {
public:
    jit_sqrt_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     ov::element::Type exec_prc = ov::element::f32);
    jit_sqrt_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);
    jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

}  // namespace ov::intel_cpu::riscv64
