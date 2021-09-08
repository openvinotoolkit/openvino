// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace preprocess {

/// \brief Information about user's input tensor. By default, it will be initialized to same data (type/shape/etc) as
/// network's input parameter User application can override particular parameters (like 'element_type') according to
/// application's data and specify appropriate conversions in pre-processing steps
///
/// \code{.cpp}
/// auto proc =
/// PrePostProcessor()
///     .input(InputInfo()
///            .tensor(InputTensorInfo()
///                    .set_element_type(ov::element::u8))
///            .preprocess(<add steps + conversion to network's input element type>)
///     );
/// \endcode
class OPENVINO_API InputTensorInfo final {
    class InputTensorInfoImpl;
    std::unique_ptr<InputTensorInfoImpl> m_impl;
    friend class InputInfo;

public:
    /// \brief Default empty constructor
    InputTensorInfo();

    /// \brief Default move constructor
    InputTensorInfo(InputTensorInfo&&) noexcept;

    /// \brief Default move assignment
    InputTensorInfo& operator=(InputTensorInfo&&) noexcept;

    /// \brief Default destructor
    ~InputTensorInfo();

    /// \brief Set element type for user's input tensor
    /// This version allows chaining for Lvalue objects
    ///
    /// \param type Element type for user's input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo& set_element_type(const ov::element::Type& type) &;

    /// \brief Set element type for user's input tensor
    /// This version allows chaining for Rvalue objects
    ///
    /// \param builder Pre-processing data for input tensor of model.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo&& set_element_type(const ov::element::Type& type) &&;
};

}  // namespace preprocess
}  // namespace ov
