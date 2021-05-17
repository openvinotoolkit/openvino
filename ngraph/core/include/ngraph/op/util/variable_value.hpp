// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
