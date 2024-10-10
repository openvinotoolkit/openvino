// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_add_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);
    jit_mul_add_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    size_t aux_vecs_count() const override;
};


class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);
    jit_subtract_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc = ov::element::f32);
    jit_multiply_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                       ov::element::Type exec_prc = ov::element::f32);
    jit_divide_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                       ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};

class jit_floor_emitter : public jit_emitter {
public:
    jit_floor_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_floor_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_ceiling_emitter : public jit_emitter {
public:
    jit_ceiling_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_ceiling_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_floor_mod_emitter : public jit_emitter {
public:
    jit_floor_mod_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                          ov::element::Type exec_prc = ov::element::f32);
    jit_floor_mod_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                          ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_mod_emitter : public jit_emitter {
public:
    jit_mod_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_mod_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_maximum_emitter : public jit_emitter {
public:
    jit_maximum_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);
    jit_maximum_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_minimum_emitter : public jit_emitter {
public:
    jit_minimum_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);
    jit_minimum_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_squared_difference_emitter : public jit_emitter {
public:
    jit_squared_difference_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                   ov::element::Type exec_prc = ov::element::f32);
    jit_squared_difference_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& n,
                                   ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_power_dynamic_emitter : public jit_emitter {
public:
    jit_power_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                              ov::element::Type exec_prc = ov::element::f32);
    jit_power_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                              ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_equal_emitter : public jit_emitter {
public:
    jit_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_not_equal_emitter : public jit_emitter {
public:
    jit_not_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                          ov::element::Type exec_prc = ov::element::f32);
    jit_not_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                          ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_greater_emitter : public jit_emitter {
public:
    jit_greater_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        ov::element::Type exec_prc = ov::element::f32);
    jit_greater_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_greater_equal_emitter : public jit_emitter {
public:
    jit_greater_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                              ov::element::Type exec_prc = ov::element::f32);
    jit_greater_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                              ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_less_emitter : public jit_emitter {
public:
    jit_less_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                     ov::element::Type exec_prc = ov::element::f32);
    jit_less_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                     ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_less_equal_emitter : public jit_emitter {
public:
    jit_less_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                           ov::element::Type exec_prc = ov::element::f32);

    jit_less_equal_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                           ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_and_emitter : public jit_emitter {
public:
    jit_logical_and_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);
    jit_logical_and_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                            ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_or_emitter : public jit_emitter {
public:
    jit_logical_or_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                           ov::element::Type exec_prc = ov::element::f32);
    jit_logical_or_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                           ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_xor_emitter : public jit_emitter {
public:
    jit_logical_xor_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);
    jit_logical_xor_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                            ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_logical_not_emitter : public jit_emitter {
public:
    jit_logical_not_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);
    jit_logical_not_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                            ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_power_static_emitter : public jit_emitter {
public:
    jit_power_static_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                             float inpPower, float inpScale, float inpShift,
                             ov::element::Type exec_prc = ov::element::f32);
    jit_power_static_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                             ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);


private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;

    float power;
    float scale;
    float shift;
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      ov::element::Type exec_prc = ov::element::f32);
    jit_prelu_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    size_t aux_vecs_count() const override;
};

class jit_sqrt_emitter : public jit_emitter {
public:
    jit_sqrt_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);
    jit_sqrt_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_negative_emitter : public jit_emitter {
public:
    jit_negative_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_exp_emitter : public jit_emitter {
public:
    jit_exp_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                    ov::element::Type exec_prc = ov::element::f32);

    jit_exp_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    bool need_vmm_mask() const { return host_isa_ != dnnl::impl::cpu::x64::avx512_core; }

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_erf_emitter : public jit_emitter {
public:
    jit_erf_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
        ov::element::Type exec_prc = ov::element::f32);

    jit_erf_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32);

    void emit_data() const override;

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(
        const std::vector<size_t> &in_vec_idxs,
        const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;

    std::unique_ptr<jit_exp_emitter> m_exp_emitter {nullptr};
};

class jit_soft_sign_emitter : public jit_emitter {
public:
    jit_soft_sign_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                          ov::element::Type exec_prc = ov::element::f32);
    jit_soft_sign_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                          ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
};

class jit_is_finite_emitter : public jit_emitter {
public:
    jit_is_finite_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t hostIsa,
                    ov::element::Type execPrc = ov::element::f32) : jit_emitter(host, hostIsa, execPrc) {
        prepare_table();
    }
    jit_is_finite_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t hostIsa, const std::shared_ptr<ov::Node>& node,
                    ov::element::Type execPrc = ov::element::f32) : jit_emitter(host, hostIsa, execPrc) {
        prepare_table();
    }

    size_t get_inputs_num() const override { return 1; };
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::f32}};
    }

protected:
    size_t aux_gprs_count() const override { return (entry_map_.empty() ? 0 : 1) + 1; }
    void register_table_entries() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_is_inf_emitter : public jit_emitter {
public:
    jit_is_inf_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t hostIsa,
                       ov::element::Type execPrc = ov::element::f32, bool detect_negative = true, bool detect_positive = true)
            : jit_emitter(host, hostIsa, execPrc), detect_negative(detect_negative), detect_positive(detect_positive) {
        prepare_table();
    }
    jit_is_inf_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t hostIsa, const std::shared_ptr<ov::Node>& node,
                       ov::element::Type execPrc = ov::element::f32): jit_emitter(host, hostIsa, execPrc) {
        prepare_table();
    }

    size_t get_inputs_num() const override { return 1; };
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::f32}};
    }

protected:
    size_t aux_gprs_count() const override { return (entry_map_.empty() ? 0 : 1) + 1; }
    void register_table_entries() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    bool detect_negative;
    bool detect_positive;
};

class jit_is_nan_emitter : public jit_emitter {
public:
    jit_is_nan_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t hostIsa,
                       ov::element::Type execPrc = ov::element::f32) : jit_emitter(host, hostIsa, execPrc) {
        prepare_table();
    }
    jit_is_nan_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t hostIsa, const std::shared_ptr<ov::Node>& node,
                       ov::element::Type execPrc = ov::element::f32) : jit_emitter(host, hostIsa, execPrc) {
        prepare_table();
    }

    size_t get_inputs_num() const override { return 1; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::f32}};
    }

protected:
    size_t aux_gprs_count() const override { return (entry_map_.empty() ? 0 : 1) + 1; }
    void register_table_entries() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_select_emitter : public jit_emitter {
public:
    jit_select_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                       ov::element::Type exec_prc = ov::element::f32);
    jit_select_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                       ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);
    size_t aux_vecs_count() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_bitwise_and_emitter : public jit_emitter {
public:
    jit_bitwise_and_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
        ov::element::Type exec_prc = ov::element::f32);
    jit_bitwise_and_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_bitwise_not_emitter : public jit_emitter {
public:
    jit_bitwise_not_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
        ov::element::Type exec_prc = ov::element::f32);
    jit_bitwise_not_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);
    size_t aux_vecs_count() const override;

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
    void register_table_entries() override;
};

class jit_bitwise_or_emitter : public jit_emitter {
public:
    jit_bitwise_or_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
        ov::element::Type exec_prc = ov::element::f32);
    jit_bitwise_or_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_bitwise_xor_emitter : public jit_emitter {
public:
    jit_bitwise_xor_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
        ov::element::Type exec_prc = ov::element::f32);
    jit_bitwise_xor_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

}   // namespace intel_cpu
}   // namespace ov
