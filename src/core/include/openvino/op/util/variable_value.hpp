// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/core_visibility.hpp"

namespace ov {
namespace op {
namespace util {
/// VariableValue stores data and state (reset flag) for a Variable,
/// and provides an interface for changing them.
class OPENVINO_API VariableValue {
public:
    using Ptr = std::shared_ptr<VariableValue>;
    /// \brief Constructs an uninitialized VariableValue.
    VariableValue() = default;

    /// \brief Constructor for VariableValue.
    /// \param value The data for Variable.
    explicit VariableValue(ngraph::HostTensorPtr value) : m_value(std::move(value)) {}

    /// \brief Constructor for VariableValue.
    /// \param value Data for Variable.
    /// \param reset The current state of the reset flag.
    VariableValue(ngraph::HostTensorPtr value, bool reset) : m_reset(reset), m_value(std::move(value)) {}

    /// \brief Sets the reset flag to a new state.
    /// \param reset The new state of the reset flag.
    void set_reset(bool reset) {
        m_reset = reset;
    }

    /// \brief Returns the current reset flag state.
    bool get_reset() const {
        return m_reset;
    }

    /// \brief Returns the current stored data.
    const ngraph::HostTensorPtr& get_value() const {
        return m_value;
    }

    /// \brief Sets new values for Variable.
    /// \param value New data for Variable.
    void set_value(const ngraph::HostTensorPtr& value) {
        m_value = value;
    }

private:
    bool m_reset = true;
    ngraph::HostTensorPtr m_value;
};
}  // namespace util
}  // namespace op
}  // namespace ov
