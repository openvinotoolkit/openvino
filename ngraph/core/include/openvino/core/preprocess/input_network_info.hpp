// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace preprocess {

/// \brief Information about network's input tensor. If all information is already included to loaded network, this info may not be needed. However it can be set to specify additional information about network, like 'layout'.
///
/// Example of usage of network 'layout':
/// Support network has input parameter with shape {1, 3, 224, 224} and user needs to resize input image to network's dimensions
/// It can be done like this
///
/// \code{.cpp}
/// <network has input parameter with shape {1, 3, 224, 224}>
/// auto proc =
/// PrePostProcessor()
///     .input(InputInfo()
///            .tensor(<input tensor info>)
///            .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
///            .network(InputNetworkInfo()
///                    .set_layout("NCHW"))
///     );
/// \endcode
class OPENVINO_API InputNetworkInfo final {
    class InputNetworkInfoImpl;
    std::unique_ptr<InputNetworkInfoImpl> m_impl;
    friend class InputInfo;

public:
    /// \brief Default empty constructor
    InputNetworkInfo();

    /// \brief Default move constructor
    InputNetworkInfo(InputNetworkInfo&&) noexcept;

    /// \brief Default move assignment
    InputNetworkInfo& operator=(InputNetworkInfo&&) noexcept;

    /// \brief Default destructor
    ~InputNetworkInfo();

    /// \brief Set layout for network's input tensor
    /// This version allows chaining for Lvalue objects
    ///
    /// \param layout Layout for network's input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputNetworkInfo& set_layout(const ov::Layout& layout) &;

    /// \brief Set layout for network's input tensor
    /// This version allows chaining for Rvalue objects
    ///
    /// \param layout Layout for network's input tensor.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputNetworkInfo&& set_layout(const ov::Layout& layout) &&;
};

}  // namespace preprocess
}  // namespace ov
