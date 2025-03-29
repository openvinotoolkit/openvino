// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov::intel_cpu {

class jit_horizon_emitter : public jit_emitter {
public:
    jit_horizon_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                        dnnl::impl::cpu::x64::cpu_isa_t isa,
                        const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 1;
    }
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::f32}};
    }

protected:
    size_t aux_vecs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    template <typename Vmm>
    void perform_op(const Vmm& vmm1, const Vmm& vmm2, const Vmm& vmm3) const;

    enum class OpType { max, sum };
    OpType m_op_type = OpType::max;
};

}  // namespace ov::intel_cpu
