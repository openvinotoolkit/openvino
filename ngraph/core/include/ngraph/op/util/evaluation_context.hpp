// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/output_vector.hpp>
#include <ngraph/variant.hpp>
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_value.hpp"

namespace ngraph
{
    class Variable;
    class VariableValue;

    using VariablePtr = std::shared_ptr<Variable>;
    using VariableValuePtr = std::shared_ptr<VariableValue>;
    using VariableMap = std::unordered_map<VariablePtr, VariableValuePtr>;

    /// VariableContext stores and manages a evaluation context for Variables.
    class VariableContext
    {
    public:
        /// \brief Constructs an uninitialized VariableContext.
        VariableContext() = default;

        /// \brief Constructor for VariableContext.
        /// \param variable_values The values associated with a particular Variables.
        explicit VariableContext(const VariableMap& variable_values)
            : m_variable_values(variable_values)
        {
        }

        /// \brief Sets the reset flags for all stored Variables to true.
        void reset_variable_context() const
        {
            for (const auto& el : m_variable_values)
            {
                el.second->set_reset(true);
            }
        }

        /// \brief Sets the new values for Variables.
        /// \param variable_values The new values associated with a particular Variable.
        void set_variable_values(const VariableMap& variable_values)
        {
            m_variable_values = variable_values;
        }

        /// \brief Changes/sets the values for Variable.
        /// \param variable New or stored Variable.
        /// \param variable_value The values associated with the variable.
        void set_variable_value(const VariablePtr& variable, const VariableValuePtr& variable_value)
        {
            m_variable_values[variable] = variable_value;
        }

        /// \brief Removes context for a particular Variable.
        /// \param variable The variable for which the context will be cleared.
        void remove_variable_value(const VariablePtr& variable)
        {
            m_variable_values.erase(variable);
        }

        /// \brief Returns the current values for Variables.
        const VariableMap& get_variable_values() const { return m_variable_values; }

        /// \brief Returns the value for specified Variable.
        VariableValuePtr get_variable_value(const VariablePtr& variable) const
        {
            auto var_value = m_variable_values.find(variable);
            if (var_value != m_variable_values.end())
            {
                return (*var_value).second;
            }
            return VariableValuePtr();
        }

    public:
        /// The values associated with a particular Variable.
        VariableMap m_variable_values;
    };

    template <>
    class VariantWrapper<VariableContext> : public VariantImpl<VariableContext>
    {
    public:
        static constexpr VariantTypeInfo type_info{"Variant::EvaluationContext::VariableContext",
                                                   0};

        const VariantTypeInfo& get_type_info() const override { return type_info; }

        explicit VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }

    private:
        using Variant::init;
        using Variant::merge;
    };

    /// EvaluationContext stores and manages a context (additional parameters, values and
    /// environment) for evaluating ngraph::function.
    class NGRAPH_API EvaluationContext
    {
    public:
        using ContextMap = std::map<std::string, std::shared_ptr<Variant>>;

        /// \brief Constructs an uninitialized EvaluationContext.
        EvaluationContext()
        {
            m_context["VariableContext"] =
                std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
        }

        /// \brief Sets a new context for Variables.
        /// \param variable_context The new context for Variables.
        void set_context(const std::string& name, const std::shared_ptr<Variant>& context)
        {
            m_context[name] = context;
        };

        /// \brief Finds and returns a context by provided name.
        std::shared_ptr<Variant> get_context_by_name(const std::string& name) const
        {
            const auto& context = m_context.find(name);
            return context != m_context.end() ? context->second : std::shared_ptr<Variant>();
        }

    private:
        ContextMap m_context;
    };
} // namespace ngraph
