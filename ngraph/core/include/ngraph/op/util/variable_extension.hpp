// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/runtime/host_tensor.hpp>
#include <utility>

namespace ngraph
{
    class NGRAPH_API VariableExtension
    {
    public:
        VariableExtension() = default;

        /// \brief Returns variable connected to this node.
        virtual std::shared_ptr<ngraph::Variable> get_variable() const { return m_variable; }

        /// \brief Sets a new variable to be connected to this node.
        ///
        /// \param variable New variable to be connected to this node.
        virtual void set_variable(const std::shared_ptr<ngraph::Variable>& variable)
        {
            m_variable = variable;
        }

        /// \brief Sets the identifier to a variable
        ///
        /// \param variable_id New identifier of the variable.
        virtual void set_variable_id(const std::string& variable_id)
        {
            m_variable->get_info().variable_id = variable_id;
        };

        /// \brief Returns the identifier of corresponding variable.
        virtual std::string get_variable_id() const = 0;

    protected:
        std::shared_ptr<ngraph::Variable> m_variable;
    };
} // namespace ngraph
