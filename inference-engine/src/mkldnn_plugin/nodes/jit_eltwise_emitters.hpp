// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/emitter.h"
#include "jit_generator.hpp"
#include "mkldnn_node.h"

namespace MKLDNNPlugin {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    size_t aux_vecs_count() const override;
};


class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_floor_mod_emitter : public jit_emitter {
public:
    jit_floor_mod_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                          InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_mod_emitter : public jit_emitter {
public:
    jit_mod_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_maximum_emitter : public jit_emitter {
public:
    jit_maximum_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_minimum_emitter : public jit_emitter {
public:
    jit_minimum_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_squared_difference_emitter : public jit_emitter {
public:
    jit_squared_difference_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                                   InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_power_dynamic_emitter : public jit_emitter {
public:
    jit_power_dynamic_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                              InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_equal_emitter : public jit_emitter {
public:
    jit_equal_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_not_equal_emitter : public jit_emitter {
public:
    jit_not_equal_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                          InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_greater_emitter : public jit_emitter {
public:
    jit_greater_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_greater_equal_emitter : public jit_emitter {
public:
    jit_greater_equal_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                              InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_less_emitter : public jit_emitter {
public:
    jit_less_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                     InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_less_equal_emitter : public jit_emitter {
public:
    jit_less_equal_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_and_emitter : public jit_emitter {
public:
    jit_logical_and_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_or_emitter : public jit_emitter {
public:
    jit_logical_or_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_xor_emitter : public jit_emitter {
public:
    jit_logical_xor_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_logical_not_emitter : public jit_emitter {
public:
    jit_logical_not_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_power_static_emitter : public jit_emitter {
public:
    jit_power_static_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                             InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode* node,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    template <mkldnn::impl::cpu::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    size_t aux_vecs_count() const override;
};

} // namespace MKLDNNPlugin
