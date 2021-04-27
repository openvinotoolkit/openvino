// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

    /// EvaluationContext stores and manages a context (additional parameters, values and
    /// environment) for evaluating ngraph::function.
    class NGRAPH_API EvaluationContext
    {
    public:
        /// \brief Constructs an uninitialized EvaluationContext.
        EvaluationContext() = default;

        /// \brief Sets a new context for Variables.
        /// \param variable_context The new context for Variables.
        void set_variable_context(const std::shared_ptr<VariableContext>& variable_context)
        {
            m_variable_context = variable_context;
        };

        /// \brief Returns the current variable context.
        const std::shared_ptr<VariableContext>& get_variable_context() const
        {
            return m_variable_context;
        }

    private:
        /// Required values for evaluating ngraph::function containing Variables.
        std::shared_ptr<VariableContext> m_variable_context = std::make_shared<VariableContext>();
    };
} // namespace ngraph
