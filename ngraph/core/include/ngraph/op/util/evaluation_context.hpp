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

#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_value.hpp"

namespace ngraph
{
    class Variable;
    class VariableValue;

    using VariablePtr = std::shared_ptr<Variable>;
    using VariableValuePtr = std::shared_ptr<VariableValue>;
    using VariableMap = std::unordered_map<VariablePtr, VariableValuePtr>;

    class VariableContext {
    public:
        void reset_variable_context() {
            for (const auto& el : m_variable_values) {
                el.second->set_reset(true);
            }
        }

        void set_variable_values(const VariableMap& variable_values) {
            m_variable_values = variable_values;
        }

        void add_variable_value(const VariablePtr& variable, const VariableValuePtr& variable_value) {
            m_variable_values[variable] = variable_value;
        }

        void remove_variable_value(const VariablePtr& variable) {
            m_variable_values.erase(variable);
        }

        const VariableMap& get_variable_values() const {
            return m_variable_values;
        }
    public:
        VariableMap m_variable_values;
    };

    class NGRAPH_API EvaluationContext
    {
    public:
        EvaluationContext() = default;

        void set_variable_context(const std::shared_ptr<VariableContext>& variable_context) {
            m_variable_context = variable_context;
        };

        const std::shared_ptr<VariableContext>& get_variable_context() const {
            return m_variable_context;
        }
    private:
        std::shared_ptr<VariableContext> m_variable_context;
    };
}
