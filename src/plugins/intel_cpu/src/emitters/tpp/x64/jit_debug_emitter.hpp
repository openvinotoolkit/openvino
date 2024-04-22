// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "jit_tpp_emitter.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface DebugTppEmitter
 * @brief The purpose of this emitter is to facilitate debugging of TPP emitters. It allows to access attributes of the
 * owned Tpp emitter at runtime, inspect source ov::Node, in/out memory before and after execution, etc.
 */
class DebugTppEmitter : public TppEmitter {
public:
    DebugTppEmitter(const ov::snippets::lowered::ExpressionPtr& expr, const std::shared_ptr<TppEmitter>& original)
            : TppEmitter(*original),
            m_original(original),
            m_compiled_kernel(m_original->get_compiled_kernel_ptr()),
            m_execute_function(m_original->get_execute_function_ptr()),
            m_source_expr(expr) {
    }

    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override {
        m_original->validate_arguments(in, out);
    };

    size_t get_inputs_num() const override { return num_kernel_args - 1; }

protected:
    static void execute_kernel_unary(const DebugTppEmitter* emitter, void *in0, void *out0) {
        OV_CPU_JIT_EMITTER_ASSERT(emitter && emitter->m_execute_function && emitter->m_compiled_kernel,
                                  "Unable to execute unary kernel");
        // Note: put a breakpoint here and analyze all the necessary debug info in runtime
        std::cout << "Running unary DebugTPPEmitter for node with name "
                  << emitter->m_source_expr->get_node()->get_friendly_name() << std::endl;
        auto f = reinterpret_cast<void(*)(uintptr_t, void*, void*)>(emitter->m_execute_function);
        f(emitter->m_compiled_kernel, in0, out0);
    }

    static void execute_kernel_binary(const DebugTppEmitter* emitter, void* in0, void* in1, void* out0) {
        OV_CPU_JIT_EMITTER_ASSERT(emitter && emitter->m_execute_function && emitter->m_compiled_kernel,
                                  "Unable to execute binary kernel");
        // Note: put a breakpoint here and analyze all the necessary debug info in runtime
        std::cout << "Running binary DebugTPPEmitter for node with name "
                  << emitter->m_source_expr->get_node()->get_friendly_name() << std::endl;
        auto f = reinterpret_cast<void(*)(uintptr_t, void*, void*, void*)>(emitter->m_execute_function);
        f(emitter->m_compiled_kernel, in0, in1, out0);
    }

    const uintptr_t get_execute_function_ptr() const override {
        // Note: num_kernel_args accounts for both input and output args
        switch (num_kernel_args) {
            case 2: return reinterpret_cast<const uintptr_t>(execute_kernel_unary);
            case 3: return reinterpret_cast<const uintptr_t>(execute_kernel_binary);
            default: OV_CPU_JIT_EMITTER_THROW("More than two arguments are not supported");
        }
    }

    const uintptr_t get_compiled_kernel_ptr() const override {
        return reinterpret_cast<const uintptr_t>(this);
    }

private:
    std::shared_ptr<TppEmitter> m_original {nullptr};
    uintptr_t m_compiled_kernel {0};
    uintptr_t m_execute_function {0};
    snippets::lowered::ExpressionPtr m_source_expr {nullptr};
};

}   // namespace intel_cpu
}   // namespace ov
