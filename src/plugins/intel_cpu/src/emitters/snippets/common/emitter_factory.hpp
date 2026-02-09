// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "cache/multi_cache.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/target_machine.hpp"

namespace ov::intel_cpu {

/**
 * @brief Factory class for creating emitter instances for snippets code generation.
 * This template class provides a flexible way to create emitters with different ISA targets
 * and customizable wrapping behavior. It supports both snippets-specific emitters and CPU plugin emitters.
 * @tparam GetHost Callable type that returns the host generator instance
 * @tparam Isa ISA type (instruction set architecture) for target platform
 * @tparam Wrap Callable type for wrapping emitter instances with additional functionality
 * @tparam GetKernelExecutorTable Callable type that returns the kernel executor table (for caching emitters)
 */
template <typename GetHost, typename Isa, typename Wrap, typename GetKernelExecutorTable = void>
class EmitterFactory {
public:
    /**
     * @brief Constructs an EmitterFactory with the specified host getter, ISA, and wrapper.
     * @param get_host Callable that provides access to the host code generator
     * @param isa The target instruction set architecture
     * @param wrap Callable to wrap created emitter instances (e.g., for logging or instrumentation)
     */
    EmitterFactory(GetHost get_host, Isa isa, Wrap wrap)
        : get_host_(std::move(get_host)),
          isa_(isa),
          wrap_(std::move(wrap)),
          get_kernel_executor_table_{},
          compiled_kernel_cache_{} {}

    /**
     * @brief Constructs an EmitterFactory with caching support.
     * @param get_host Callable that provides access to the host code generator
     * @param isa The target instruction set architecture
     * @param wrap Callable to wrap created emitter instances
     * @param get_kernel_executor_table Callable that returns the current kernel executor table
     * @param compiled_kernel_cache Weak pointer to the compiled kernel cache
     */
    template <typename T = GetKernelExecutorTable, typename = std::enable_if_t<!std::is_void_v<T>>>
    EmitterFactory(GetHost get_host,
                   Isa isa,
                   Wrap wrap,
                   T get_kernel_executor_table,
                   MultiCacheWeakPtr compiled_kernel_cache)
        : get_host_(std::move(get_host)),
          isa_(isa),
          wrap_(std::move(wrap)),
          get_kernel_executor_table_(std::move(get_kernel_executor_table)),
          compiled_kernel_cache_(std::move(compiled_kernel_cache)) {}

    /**
     * @brief Creates a jitters_value for emitters that are constructed from a snippets expression.
     * This is the simple form for emitters that don't require additional constructor arguments
     * beyond get_host(), isa, and expr.
     * @tparam Emitter The emitter class type to instantiate
     * @return A jitters_value containing factory and precision query functions
     */
    template <typename Emitter>
    [[nodiscard]] ov::snippets::jitters_value from_expr() const {
        return {[get_host = get_host_, isa = isa_, wrap = wrap_](
                    const ov::snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<ov::snippets::Emitter> {
                    auto emitter = std::make_shared<Emitter>(get_host(), isa, expr);
                    return wrap(emitter, expr);
                },
                [](const std::shared_ptr<ov::Node>& n) -> std::set<ov::element::TypeVector> {
                    return Emitter::get_supported_precisions(n);
                }};
    }

    /**
     * @brief Creates a jitters_value for emitters that require kernel executor table and cache.
     * This method is used for emitters that support runtime recompilation and caching
     * (e.g., BRGEMM emitters). It uses the kernel_executor_table and compiled_kernel_cache
     * members that were provided during factory construction.
     * @tparam Emitter The emitter class type to instantiate
     * @return A jitters_value containing factory and precision query functions
     */
    template <typename Emitter, typename T = GetKernelExecutorTable, typename = std::enable_if_t<!std::is_void_v<T>>>
    [[nodiscard]] ov::snippets::jitters_value from_expr_cached() const {
        OPENVINO_ASSERT(!compiled_kernel_cache_.expired(), "compiled_kernel_cache_ is expired in from_expr_cached");
        return {[get_host = get_host_,
                 isa = isa_,
                 wrap = wrap_,
                 get_kernel_table = get_kernel_executor_table_,
                 compiled_kernel_cache = compiled_kernel_cache_](
                    const ov::snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<ov::snippets::Emitter> {
                    auto emitter =
                        std::make_shared<Emitter>(get_host(), isa, expr, get_kernel_table(), compiled_kernel_cache);
                    return wrap(emitter, expr);
                },
                [](const std::shared_ptr<ov::Node>& n) -> std::set<ov::element::TypeVector> {
                    return Emitter::get_supported_precisions(n);
                }};
    }

    /**
     * @brief Creates a jitters_value for emitters that are constructed from an OpenVINO node.
     * This method generates a factory function for emitters that only require the host, ISA,
     * and the node from the expression. Unlike from_expr(), this does not apply the wrap_ callable.
     * @tparam Emitter The emitter class type to instantiate
     * @return A jitters_value containing factory and precision query functions
     */
    template <typename Emitter>
    [[nodiscard]] ov::snippets::jitters_value from_node() const {
        return {[get_host = get_host_, isa = isa_](
                    const ov::snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<ov::snippets::Emitter> {
                    return std::make_shared<Emitter>(get_host(), isa, expr->get_node());
                },
                [](const std::shared_ptr<ov::Node>& n) -> std::set<ov::element::TypeVector> {
                    return Emitter::get_supported_precisions(n);
                }};
    }

    /**
     * @brief Creates a jitters_value for operations that are decomposed into low-level expressions.
     * This method returns a factory that produces nullptr (indicating no direct emitter implementation)
     * while still advertising the supported precisions for the operation. This is used for
     * high-level operations that are decomposed into lower-level expressions during compilation,
     * where the actual code generation happens at the decomposed expression level.
     * @param supported_precisions The set of precision combinations supported by this operation
     * @return A jitters_value with a null emitter factory and the provided precision information
     */
    [[nodiscard]] static ov::snippets::jitters_value undefined(std::set<ov::element::TypeVector> supported_precisions) {
        return {[](const ov::snippets::lowered::ExpressionPtr&) -> std::shared_ptr<ov::snippets::Emitter> {
                    return nullptr;
                },
                [supported_precisions = std::move(supported_precisions)](
                    const std::shared_ptr<ov::Node>&) -> std::set<ov::element::TypeVector> {
                    return supported_precisions;
                }};
    }

private:
    GetHost get_host_;
    Isa isa_;
    Wrap wrap_;
    std::conditional_t<std::is_void_v<GetKernelExecutorTable>, std::monostate, GetKernelExecutorTable>
        get_kernel_executor_table_;
    std::conditional_t<std::is_void_v<GetKernelExecutorTable>, std::monostate, MultiCacheWeakPtr>
        compiled_kernel_cache_;
};

template <typename GetHost, typename Isa, typename Wrap>
EmitterFactory(GetHost, Isa, Wrap) -> EmitterFactory<GetHost, Isa, Wrap, void>;

template <typename GetHost, typename Isa, typename Wrap, typename GetKernelExecutorTable>
EmitterFactory(GetHost, Isa, Wrap, GetKernelExecutorTable, MultiCacheWeakPtr)
    -> EmitterFactory<GetHost, Isa, Wrap, GetKernelExecutorTable>;

}  // namespace ov::intel_cpu
