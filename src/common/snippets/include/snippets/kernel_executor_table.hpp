// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/util/pp.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include <typeinfo>
#if defined(SNIPPETS_DEBUG_CAPS) && !defined(_WIN32)
#include <cxxabi.h>
#endif
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

        /*** Return deep copy of the config */
        virtual std::unique_ptr<GenericConfig> get_clone_ptr() const = 0;

        /*** Compute hash for fast comparison operations or caching support */
        virtual size_t hash() const = 0;

        virtual ~GenericConfig() = default;
        /** serialize config for debug purposes */
#ifdef SNIPPETS_DEBUG_CAPS
        virtual std::string to_string() const = 0;
#endif
    };
    /**
    * @brief Update current kernel config in accordance with the passed expression. Corresponding kernel is recompiled if necessary.
     * This method should be called to update KernelExecutor based on runtime info (e.g. shapes) available through expression ptr
    */
    virtual void update_by_expression(const lowered::ExpressionPtr& expr, const lowered::LinearIRCPtr& linear_ir) = 0;
    /**
    * @brief Replace current kernel config with the provided value. Corresponding kernel is recompiled if necessary.
     * This method should be called to restore a saved state of the executor, that was configured using update_by_expression().
    */
    virtual void update_by_config(const GenericConfig& new_config) = 0;

    virtual const GenericConfig& get_config() const = 0;
    /** serialize for debug purposes */
#ifdef SNIPPETS_DEBUG_CAPS
    virtual std::string to_string() const = 0;
#endif
    virtual ~KernelExecutorBase() = default;

private:
    KernelExecutorBase() = default;
    template<typename Conf, typename KernelType,
            typename std::enable_if<std::is_base_of<GenericConfig, Conf>::value, bool>::type> friend class KernelExecutor;
};

template<typename Conf, typename KernelType,
         typename std::enable_if<std::is_base_of<KernelExecutorBase::GenericConfig, Conf>::value, bool>::type = true>
class KernelExecutor : public KernelExecutorBase {
public:
    explicit KernelExecutor(Conf c) : KernelExecutorBase(), m_config{std::move(c)} {}

    // Note: override when final is redundant, but needed to avoid warnings on some compilers
    void update_by_expression(const lowered::ExpressionPtr& expr, const lowered::LinearIRCPtr& linear_ir) override final { // NOLINT
        update_config(expr, linear_ir, m_config);
        OPENVINO_ASSERT(m_config.is_completed(), "Failed to update kernel config in update_by_expression");
        update_kernel(m_config, m_kernel);
        OPENVINO_ASSERT(m_kernel, "Failed to compile kernel executor");
    }
    void update_by_config(const GenericConfig& new_config) override final { // NOLINT
        if (m_config.hash() == new_config.hash())
            return;
        const auto& new_ptr = dynamic_cast<const Conf*>(&new_config);
        OPENVINO_ASSERT(new_config.is_completed() && new_ptr, "Failed to update kernel config in get_config");
        m_config = *new_ptr;
        update_kernel(m_config, m_kernel);
        OPENVINO_ASSERT(m_kernel, "Failed to compile kernel executor");
    }
    const GenericConfig& get_config() const override { return m_config; }
    std::shared_ptr<const KernelType> get_kernel() const { return m_kernel; }
#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override {
        std::string type_name = typeid(KernelType).name();
#ifndef _WIN32
        int status;
        std::unique_ptr<char, void (*)(void*)> demangled_name(
                abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status),
                std::free);
        type_name = demangled_name.get();
#endif
        return  "KernelExecutorType: " + std::string(type_name) + " KernelConfig: " + m_config.to_string();
    }
#endif

protected:
    /*** Updates stored kernel config based on runtime info from expression (e.g. new input shapes). */
    virtual void update_config(const lowered::ExpressionPtr& expr, const lowered::LinearIRCPtr& linear_ir, Conf& config) const = 0;
    /*** Updates stored kernel in accordance with the passed config. Recompilation of the kernel is
     * performed if necessary. */
    virtual void update_kernel(const Conf& c, std::shared_ptr<KernelType>& kernel) const = 0;

private:
    /** Contains all the necessary information to compile a desired kernel*/
    Conf m_config {};
    /** Stores pointer to compiled kernel since the last update_kernel() call */
    std::shared_ptr<KernelType> m_kernel = nullptr;
};

class KernelExecutorTable {
public:
    /*** Register KernelExecutor in the KernelExecutorTable so it can be later updated in runtime. */
    template<typename T, class ...C,
            typename std::enable_if<std::is_base_of<KernelExecutorBase, T>::value, bool>::type = true>
    std::shared_ptr<T> register_kernel(const lowered::ExpressionPtr& expr, C... args) {
        const auto& instance = std::make_shared<T>(args...);
        OPENVINO_ASSERT(m_table.insert({expr->get_exec_num(), instance}).second, "This expression execution number already has an alterable kernel");
        return instance;
    }

    const std::shared_ptr<KernelExecutorBase>& get_kernel_executor(const lowered::ExpressionPtr& expr) const {
        return get_kernel_executor(expr->get_exec_num());
    }
    const std::shared_ptr<KernelExecutorBase>& get_kernel_executor(double expr_exec_num) const {
        OPENVINO_ASSERT(m_table.count(expr_exec_num), "This expression execution number doesn't have a registered kernel executor");
        return m_table.at(expr_exec_num);
    }

    /*** Updates every registered KernelExecutor in accordance with the corresponding expression */
    void update_state(const lowered::LinearIRCPtr& linear_ir) const {
        for (const auto& expr : *linear_ir) {
            const auto& found = m_table.find(expr->get_exec_num());
            if (found != m_table.end()) {
                found->second->update_by_expression(expr, linear_ir);
            }
        }
    }

    /*** Returns lambda function that contains current state of the table, and restores this state when called  */
    std::function<void()> get_state_reset() {
        auto current_state = get_state();
        return [OV_CAPTURE_CPY_AND_THIS]() { reset_state(current_state); };
    }

    virtual ~KernelExecutorTable() = default;

protected:
    std::unordered_map<double, std::shared_ptr<KernelExecutorBase>> m_table {};

    typedef std::vector<std::pair<double, std::shared_ptr<const KernelExecutorBase::GenericConfig>>> ExecTableState;

    /*** Restore the table state previously obtained by get_state() */
    void reset_state(const ExecTableState& state);

    /*** Return cumulative state of all the executors in the table. The returned ExecTableState object can be passed to reset_state */
    ExecTableState get_state() const;
};

using KernelExecutorTablePtr = std::shared_ptr<KernelExecutorTable>;


} // namespace snippets
} // namespace ov
