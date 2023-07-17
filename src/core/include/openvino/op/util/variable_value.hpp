// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED_VARIABLE_VALUE
#endif

#include "ngraph/runtime/host_tensor.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED_VARIABLE_VALUE
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED_VARIABLE_VALUE
#endif

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace util {
/// VariableValue stores data and state (reset flag) for a Variable,
/// and provides an interface for changing them.
class OPENVINO_API VariableValue {
public:
    using Ptr = std::shared_ptr<VariableValue>;
    /// \brief Constructs an uninitialized VariableValue.
    VariableValue();

    /// \brief Constructor for VariableValue.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    /// \param value The data for Variable.
    OPENVINO_DEPRECATED(
        "This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor instead.")
    explicit VariableValue(ngraph::HostTensorPtr value);

    /// \brief Constructor for VariableValue.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    /// \param value Data for Variable.
    /// \param reset The current state of the reset flag.
    OPENVINO_DEPRECATED(
        "This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor instead.")
    VariableValue(ngraph::HostTensorPtr value, bool reset);

    /// \brief Returns the current stored data.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    OPENVINO_DEPRECATED("This method is deprecated and will be removed in 2024.0 release. Please get_state() instead.")
    ngraph::HostTensorPtr get_value() const;

    /// \brief Sets new values for Variable.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    /// \param value New data for Variable.
    OPENVINO_DEPRECATED(
        "This method is deprecated and will be removed in 2024.0 release. Please use set_state() instead.")
    void set_value(const ngraph::HostTensorPtr& value);

    /// \brief Sets the reset flag to a new state.
    /// \param reset The new state of the reset flag.
    void set_reset(bool reset);

    /// \brief Returns the current reset flag state.
    bool get_reset() const;

    explicit VariableValue(const ov::Tensor& value);

    /// \brief Constructor for VariableValue.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    /// \param value Data for Variable.
    /// \param reset The current state of the reset flag.
    VariableValue(const ov::Tensor& value, bool reset);

    /// \brief Returns the current stored data.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    const ov::Tensor& get_state() const;

    /// \brief Sets new values for Variable.
    /// \deprecated This method is deprecated and will be removed in 2024.0 release. Please use method with ov::Tensor
    /// instead
    /// \param value New data for Variable.
    void set_state(const ov::Tensor& value);

private:
    bool m_reset = true;
    ov::Tensor m_value;
};
}  // namespace util
}  // namespace op
}  // namespace ov
