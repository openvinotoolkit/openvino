// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/expression.hpp"

namespace ov {
namespace snippets {

/**
* @brief Base class for all kernel executors. This class should not be instantiated directly.
 * Derive from KernelExecutor<> to create desired executor implementations.
*/
class KernelExecutorBase {
public:
    class GenericConfig {
    public:
        /**
        * @brief Returns true if the config specifies all the parameters necessary for kernel compilation.
         * Configs for static kernels should be completed on code emission stage,
         * while dynamic kernels will be completed only in runtime, when all the shapes are known.
        */
        virtual bool is_completed() const = 0;
        virtual ~GenericConfig() = default;
    };
    virtual ~KernelExecutorBase() = default;

private:
    KernelExecutorBase() = default;
    template<typename Conf, typename KernelType,
            typename std::enable_if<std::is_base_of<GenericConfig, Conf>::value, bool>::type> friend class KernelExecutor;
};

template<typename Conf, typename KernelType,
         typename std::enable_if<std::is_base_of<KernelExecutorBase::GenericConfig, Conf>::value, bool>::type = true>
class KernelExecutor : public snippets::KernelExecutorBase {
public:
    explicit KernelExecutor(std::shared_ptr<Conf> c) : KernelExecutorBase(), m_config{std::move(c)} {}
    /**
    * @brief check current config and recompile kernel if necessary. Use kernel caching to avoid redundant recompilations.
     * This method must be called only for complete configs. It's the user responsibility to check is_completed() before calling.
    */
    virtual void update_kernel()  = 0;
protected:
    /**
    * @brief Takes shared_ptr to compilation config, returns shared_ptr to compiled kernel.
     * Should be called only if actual compilation is required. Kernel caching must be implemented in update_kernel().
    */
    virtual std::shared_ptr<KernelType> compile_kernel(const std::shared_ptr<Conf>& c) const = 0;
    /** Contains all the necessary information to compile a desired kernel*/
    std::shared_ptr<Conf> m_config = nullptr;
    /** Stores pointer to compiled kernel since the last update_kernel() call */
    std::shared_ptr<KernelType> m_kernel = nullptr;
};

class KernelExecutorTable {
public:
    template<typename T, class ...C,
            typename std::enable_if<std::is_base_of<KernelExecutorBase, T>::value, bool>::type = true>
    std::shared_ptr<T> register_kernel(const snippets::lowered::ExpressionPtr& expr, C... args) {
        OPENVINO_ASSERT(!m_table.count(expr), "This expression already has an alterable kernel");
        const auto& instance = std::make_shared<T>(args...);
        m_table[expr] = instance;
        return instance;
    }
    std::shared_ptr<KernelExecutorBase> get_kernel_executor(const snippets::lowered::ExpressionPtr& expr) const {
        OPENVINO_ASSERT(m_table.count(expr), "This expression doesn't have a registered kernel executor");
        return m_table.at(expr);
    }
    virtual ~KernelExecutorTable() = default;

protected:
    std::unordered_map<snippets::lowered::ExpressionPtr, std::shared_ptr<KernelExecutorBase>> m_table{};
};

using KernelExecutorTablePtr = std::shared_ptr<KernelExecutorTable>;


} // namespace snippets
} // namespace ov
