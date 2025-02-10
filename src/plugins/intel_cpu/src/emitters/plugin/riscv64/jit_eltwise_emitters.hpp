// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc = ov::element::f32);
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
};

class jit_clamp_emitter : public jit_emitter {
public:
    jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, float min, float max, const ov::element::Type exec_prc = ov::element::f32);
    jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    bool need_table() const override;
    const void* get_table() const override;

    float min = 0.f;
    float max = 0.f;
};

class jit_div_emitter : public jit_emitter {
public:
    jit_div_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc = ov::element::f32);
    jit_div_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
};

class jit_mul_emitter : public jit_emitter {
public:
    jit_mul_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc = ov::element::f32);
    jit_mul_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc = ov::element::f32);
    jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
};

class jit_relu_emitter : public jit_emitter {
public:
    jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, float alpha, const ov::element::Type exec_prc = ov::element::f32);
    jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
    bool need_table() const override;
    const void* get_table() const override;

    float alpha = 0.f;
};

class jit_sub_emitter : public jit_emitter {
public:
    jit_sub_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc = ov::element::f32);
    jit_sub_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;
};

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov
