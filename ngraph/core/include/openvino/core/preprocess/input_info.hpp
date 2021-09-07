// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"

namespace ov {
namespace preprocess {

/// \brief Class holding preprocessing information for one input
/// API has Builder-like style to allow chaining calls in client's code, like
/// \code{.cpp}
/// auto proc = PrePostProcessor().input(InputInfo().tensor(...).preprocess(...);
/// \endcode
class OPENVINO_API InputInfo final {
    class InputInfoImpl;
    std::unique_ptr<InputInfoImpl> m_impl;
    friend class PrePostProcessor;

public:
    /// \brief Empty constructor. Should be used only if network will have only one input
    InputInfo();

    /// \brief Information about info for particular input index of model
    ///
    /// \param input_index Index to address specified input parameter of model
    InputInfo(size_t input_index);

    /// \brief Default move constructor
    InputInfo(InputInfo&&) noexcept;

    /// \brief Default move assignment operator
    InputInfo& operator=(InputInfo&&) noexcept;

    /// \brief Default destructor
    ~InputInfo();

    /// \brief Set input tensor information for input - Lvalue version
    ///
    /// \param builder Input tensor information.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputInfo& tensor(InputTensorInfo&& builder) &;

    /// \brief Set input tensor information for input - Rvalue version
    ///
    /// \param builder Input tensor information.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputInfo&& tensor(InputTensorInfo&& builder) &&;

    /// \brief Set preprocessing operations for input - Lvalue version
    ///
    /// \param builder Preprocessing operations.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputInfo& preprocess(PreProcessSteps&& builder) &;

    /// \brief Set preprocessing operations for input - Rvalue version
    ///
    /// \param builder Preprocessing operations.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputInfo&& preprocess(PreProcessSteps&& builder) &&;
};

}  // namespace preprocess
}  // namespace ov
