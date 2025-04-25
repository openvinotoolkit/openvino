// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "emitters/tpp/common/kernel_executors/brgemm.hpp"
#include "jit_tpp_emitter.hpp"

namespace ov::intel_cpu {

class BrgemmTppEmitter : public TppEmitter {
public:
    BrgemmTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr,
                     const snippets::KernelExecutorTablePtr& kernel_table,
                     const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);

    size_t get_inputs_num() const override {
        return 2;
    }
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

    const uintptr_t get_execute_function_ptr() const override;
    const uintptr_t get_compiled_kernel_ptr() const override;

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    static void validate_subtensors(const VectorDims& in_0, const VectorDims& in_1, const VectorDims& out_0);

private:
    std::shared_ptr<tpp::BrgemmKernelExecutor> m_kernel_executor = nullptr;
};

}  // namespace ov::intel_cpu
