//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <ngraph/runtime/host_tensor.hpp>
#include <utility>

namespace ngraph
{
    class NGRAPH_API Memory
    {
    public:
        Memory() = default;

        /// \brief Returns variable connected to this node.
        virtual std::shared_ptr<ngraph::Variable> get_variable() const { return m_variable; }
        /// \brief Sets a new variable to be connected to this node.
        ///
        /// \param variable New variable to be connected to this node.
        virtual void set_variable(const std::shared_ptr<ngraph::Variable>& variable)
        {
            m_variable = variable;
        }

        /// \brief Sets the identifier of corresponding variable
        ///
        /// \param variable_id New identifier of the variable.
        virtual void set_variable_id(const std::string& variable_id){};

        /// \brief Returns the identifier of corresponding variable.
        virtual std::string get_variable_id() const = 0;

    protected:
        std::shared_ptr<ngraph::Variable> m_variable;
    };
}
