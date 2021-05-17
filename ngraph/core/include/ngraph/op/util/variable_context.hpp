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
    using VariableValuePtr = std::shared_ptr<VariableValue>;
    using VariableMap = std::unordered_map<VariablePtr, VariableValuePtr>;

    /// VariableContext stores and manages a evaluation context for Variables.
    class NGRAPH_API VariableContext
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

    private:
        /// The values associated with a particular Variable.
        VariableMap m_variable_values;
    };

    template <>
    class NGRAPH_API VariantWrapper<VariableContext> : public VariantImpl<VariableContext>
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

} // namespace ngraph
