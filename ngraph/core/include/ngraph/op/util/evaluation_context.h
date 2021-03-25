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
#include "ngraph/op/util/variable_value.h"

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
            for (const auto& el : m_variable_context) {
                el.second->set_reset(true);
            }
        }

        void add_variable_context(const VariableMap& variable_context_to_add) {
            m_variable_context.insert(variable_context_to_add.begin(), variable_context_to_add.end());
        }

        void add_variable_value(const VariablePtr& variable, const VariableValuePtr& variable_value) {
            m_variable_context[variable] = variable_value;
        }

        const VariableMap& get_context() const {
            return m_variable_context;
        }
    public:
        VariableMap m_variable_context;
    };
    // Base class (interface) e.g. Context, methods reset, set, add, get
    // class EvaluationContext : public Context
    // class VariableContext : public Context
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
