// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu {
class ScalarTppEmitter : public jit_emitter {
public:
    ScalarTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override {
        return 0;
    }
    size_t aux_gprs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
};

}  // namespace ov::intel_cpu
