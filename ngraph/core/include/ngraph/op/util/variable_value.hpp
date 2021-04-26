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
    /// VariableValue stores data and state (reset flag) for a Variable,
    /// and provides an interface for changing them.
    class NGRAPH_API VariableValue
    {
    public:
        /// \brief Constructs an uninitialized VariableValue.
        VariableValue() = default;

        /// \brief Constructor for VariableValue.
        /// \param value The data for Variable.
        explicit VariableValue(HostTensorPtr value)
            : m_value(std::move(value))
        {
        }

        /// \brief Constructor for VariableValue.
        /// \param value Data for Variable.
        /// \param reset The current state of the reset flag.
        VariableValue(HostTensorPtr value, bool reset)
            : m_reset(reset)
            , m_value(std::move(value))
        {
        }

        /// \brief Sets the reset flag to a new state.
        /// \param reset The new state of the reset flag.
        void set_reset(bool reset) { m_reset = reset; }

        /// \brief Returns the current reset flag state.
        bool get_reset() const { return m_reset; }

        /// \brief Returns the current stored data.
        const HostTensorPtr& get_value() const { return m_value; }

        /// \brief Sets new values for Variable.
        /// \param value New data for Variable.
        void set_value(const HostTensorPtr& value) { m_value = value; }

    private:
        bool m_reset = true;
        HostTensorPtr m_value;
    };
    using VariableValuePtr = std::shared_ptr<VariableValue>;
} // namespace ngraph
