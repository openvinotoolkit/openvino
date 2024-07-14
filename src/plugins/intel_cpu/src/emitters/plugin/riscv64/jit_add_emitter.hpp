// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                    const ov::element::Type exec_prc = ov::element::f32);

    jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                    const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
