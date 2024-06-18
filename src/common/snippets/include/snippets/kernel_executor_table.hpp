// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
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
        virtual std::shared_ptr<GenericConfig> clone() const = 0;

        /*** Compute hash for fast comparison operations or caching support */
        virtual size_t hash() const = 0;

        bool operator==(const GenericConfig& rhs) const { return hash() == rhs.hash(); }
        bool operator!=(const GenericConfig& rhs) const { return hash() != rhs.hash(); }

        virtual ~GenericConfig() = default;
        /** serialize config for debug purposes */
#ifdef SNIPPETS_DEBUG_CAPS
        virtual std::string to_string() const = 0;
#endif
    };
    /**
    * @brief update current kernel config and recompile kernel if necessary.
     * This method should be called to update KernelExecutor based on runtime info (e.g. shapes) available through expression ptr
    */
    virtual void update_by_expression(const ov::snippets::lowered::ExpressionPtr& expr) = 0;
    virtual void update_by_config(const std::shared_ptr<const GenericConfig>& new_config) = 0;

    virtual std::shared_ptr<const GenericConfig> get_config() const = 0;
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
class KernelExecutor : public snippets::KernelExecutorBase {
public:
    explicit KernelExecutor(std::shared_ptr<Conf> c) : KernelExecutorBase(), m_config{std::move(c)} {}

    // Note: override when final is redundant, but needed to avoid warnings on some compilers
    void update_by_expression(const ov::snippets::lowered::ExpressionPtr& expr) override final { // NOLINT
        update_config(expr, m_config);
        OPENVINO_ASSERT(m_config && m_config->is_completed(), "Failed to update kernel config in update_by_expression");
        update_kernel(m_config, m_kernel);
        OPENVINO_ASSERT(m_kernel, "Failed to compile kernel executor");
    }
    void update_by_config(const std::shared_ptr<const GenericConfig>& new_config) override final { // NOLINT
        if (*m_config == *new_config)
            return;
        m_config = std::static_pointer_cast<Conf>(std::const_pointer_cast<GenericConfig>(new_config));
        OPENVINO_ASSERT(m_config && m_config->is_completed(), "Failed to update kernel config in get_config");
        update_kernel(m_config, m_kernel);
        OPENVINO_ASSERT(m_kernel, "Failed to compile kernel executor");
    }
    std::shared_ptr<const GenericConfig> get_config() const override { return m_config; }
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
        return  "KernelExecutorType: " + std::string(type_name) + " KernelConfig: " + m_config->to_string();
    }
#endif

protected:
    /*** Updates stored kernel config based on runtime info from expression (e.g. new input shapes). */
    virtual void update_config(const ov::snippets::lowered::ExpressionPtr& expr, std::shared_ptr<Conf>& config) const = 0;
    /*** Updates stored kernel in accordance with the passed config. Recompilation of the kernel is
     * performed only if necessary, otherwise an appropriate kernel is retrieved from cache. */
    virtual void update_kernel(const std::shared_ptr<const Conf>& c, std::shared_ptr<KernelType>& kernel) const = 0;

private:
    /** Contains all the necessary information to compile a desired kernel*/
    std::shared_ptr<Conf> m_config = nullptr;
    /** Stores pointer to compiled kernel since the last update_kernel() call */
    std::shared_ptr<KernelType> m_kernel = nullptr;
};

class KernelExecutorTable {
public:
    typedef std::vector<std::pair<snippets::lowered::ExpressionPtr, std::shared_ptr<const KernelExecutorBase::GenericConfig>>> ExecTableState;
    /*** Register KernelExecutor in the KernelExecutorTable so it can be later updated in runtime. */
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
    /*** Update every registered KernelExecutor in accordance with the corresponding expression */
    void update_state() const {
        for (const auto& record : m_table)
            record.second->update_by_expression(record.first);
    }
    /*** Return cumulative state of all the executors in the table. The returned ExecTableState object can be passed to reset_state */
    ExecTableState get_state() const {
        ExecTableState result;
        // Note: we need to clone configs when saving the state, since the configs still stored in the table can
        // be modified e.g. by calling update_by_expression();
        for (const auto& record : m_table)
            result.emplace_back(std::make_pair(record.first, record.second->get_config()->clone()));
        return result;
    }
    /*** Restore the table state previously obtained by get_state() */
    void reset_state(const ExecTableState& state) {
        OPENVINO_ASSERT(state.size() == m_table.size(), "Invalid state in restore_state: size mismatch");
        auto state_it = state.begin();
        for (const auto& table_record : m_table) {
            const auto& state_record = *state_it++;
            OPENVINO_ASSERT(table_record.first == state_record.first, "Invalid state in restore_state: expressions mismatch");
            table_record.second->update_by_config(state_record.second);
        }
    }

    /**
    * @brief Replace originally registered ExpressionPtr with a new value.
     * Note that code emission is performed on a copy of LIR, so all expression pointers visible from emitters won't
     * be accessible from RuntimeConfigurator. In order to replace these cloned ExpressionPtrs with the original ones,
     * we need to call this method.
    */
    bool replace_reference_expression(const snippets::lowered::ExpressionPtr& from, const snippets::lowered::ExpressionPtr& to) {
        const auto& found = m_table.find(from);
        if (found != m_table.end()) {
            OPENVINO_ASSERT(m_table.count(to) == 0, "Attempt to replace a value that is already in the KernelExecutorTable");
            m_table.insert({to, found->second});
            m_table.erase(found);
            return true;
        }
        return false;
    }
    virtual ~KernelExecutorTable() = default;

protected:
    std::unordered_map<snippets::lowered::ExpressionPtr, std::shared_ptr<KernelExecutorBase>> m_table{};
};

using KernelExecutorTablePtr = std::shared_ptr<KernelExecutorTable>;


} // namespace snippets
} // namespace ov
