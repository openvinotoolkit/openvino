// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"

namespace ov {
namespace preprocess {

/// \brief Information about network's output tensor. If all information is already included to loaded network, this
/// info may not be needed. However it can be set to specify additional information about network, like 'layout'.
///
/// Example of usage of network 'layout':
/// Support network has output parameter with shape {1, 3, 224, 224} and `NHWC` layout. User may need to transpose
/// output picture to interleaved format {1, 224, 224, 3}. This can be done with the following code
///
/// \code{.cpp}
/// <network has output parameter with shape {1, 3, 224, 224}>
/// auto proc =
/// PrePostProcessor()
///     .output(OutputInfo()
///            .network(OutputNetworkInfo().set_layout("NCHW")
///            .preprocess(PostProcessSteps().convert_layout("NHWC")))
///     );
/// \endcode
class OPENVINO_API OutputNetworkInfo final {
    class OutputNetworkInfoImpl;
    std::unique_ptr<OutputNetworkInfoImpl> m_impl;
    friend class OutputInfo;

public:
    /// \brief Default empty constructor
    OutputNetworkInfo();

    /// \brief Default move constructor
    OutputNetworkInfo(OutputNetworkInfo&&) noexcept;

    /// \brief Default move assignment
    OutputNetworkInfo& operator=(OutputNetworkInfo&&) noexcept;

    /// \brief Default destructor
    ~OutputNetworkInfo();

    /// \brief Set layout for network's output tensor
    /// This version allows chaining for Lvalue objects
    ///
    /// \param layout Layout for network's output tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputNetworkInfo& set_layout(const ov::Layout& layout) &;

    /// \brief Set layout for network's output tensor
    /// This version allows chaining for Rvalue objects
    ///
    /// \param layout Layout for network's output tensor.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    OutputNetworkInfo&& set_layout(const ov::Layout& layout) &&;
};

}  // namespace preprocess
}  // namespace ov
