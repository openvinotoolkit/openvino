// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include "snippets/op/loop.hpp"


namespace ov {
namespace intel_cpu {

class jit_loop_begin_emitter : public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;
    // todo: it is purely virtual in the base class, but do we need it?
    size_t get_inputs_num() const override {return 0;}

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::shared_ptr<snippets::op::LoopEnd> get_loop_end(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    bool evaluate_once = false;
    size_t work_amount = 0; // need to store work_amount explicitly, since two loops can work on the same dim (e.g. vector + scalar)
};

class jit_loop_end_emitter : public jit_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;
    // todo: it is purely virtual in the base class, but do we need it?
    size_t get_inputs_num() const override {return 0;}

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    std::shared_ptr<snippets::op::LoopEnd> loop_end;

    size_t num_inputs = 0;
    size_t num_outputs = 0;
    // keep data_size int64_t to avoid conversion to size_t (and overflow) when multiplied by negative increments or offsets
    std::vector<int64_t> io_data_size {};
    int64_t wa_increment = 0;
    int64_t work_amount = 0;
    bool evaluate_once = false;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
};

}   // namespace intel_cpu
}   // namespace ov