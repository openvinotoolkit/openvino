// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

class Node;

namespace preprocess {

/// \brief Postprocessing steps. Each step typically intends adding of some operation to output parameter
/// User application can specify sequence of postprocessing steps in a builder-like manner
/// \code{.cpp}
/// auto proc = PrePostProcessor()
///     .output(OutputInfo()
///             .postprocess(PostProcessSteps()
///                        .convert_element_type(element::u8)))
///     );
/// \endcode
class OPENVINO_API PostProcessSteps final {
    class PostProcessStepsImpl;
    std::unique_ptr<PostProcessStepsImpl> m_impl;
    friend class OutputInfo;

public:
    /// \brief Default empty constructor
    PostProcessSteps();

    /// \brief Default move constructor
    PostProcessSteps(PostProcessSteps&&) noexcept;

    /// \brief Default move assignment operator
    PostProcessSteps& operator=(PostProcessSteps&&) noexcept;

    /// \brief Default destructor
    ~PostProcessSteps();

    /// \brief Add convert element type post-process operation - Lvalue version
    ///
    /// \param type Desired type of output. If not specified, type will be obtained from 'tensor' output information
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps& convert_element_type(const ov::element::Type& type = {}) &;

    /// \brief Add convert element type post-process operation - Rvalue version
    ///
    /// \param type Desired type of output. If not specified, type will be obtained from 'tensor' output information
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps&& convert_element_type(const ov::element::Type& type = {}) &&;

    /// \brief Add 'convert layout' operation to specified layout - Lvalue version.
    ///
    /// \details Adds appropriate 'transpose' operation between network layout and user's desired layout.
    /// Current implementation requires source and destination layout to have same number of dimensions
    ///
    /// \example Example: when network data has output in 'NCHW' layout ([1, 3, 224, 224]) but user needs
    /// interleaved output image ('NHWC', [1, 224, 224, 3]). Post-processing may look like this:
    ///
    /// \code{.cpp} auto proc =
    /// PrePostProcessor()
    ///     .output(OutputInfo()
    ///            .network(OutputTensorInfo().set_layout("NCHW")) // Network output is NCHW
    ///            .postprocess(PostProcessSteps()
    ///                        .convert_layout("NHWC")) // User needs output as NHWC
    ///     );
    /// \endcode
    ///
    /// \param dst_layout New layout after conversion. If not specified - destination layout is obtained from
    /// appropriate tensor output properties.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    PostProcessSteps& convert_layout(const Layout& dst_layout = {}) &;

    /// \brief Add convert_layout operation to network dimensions - Rvalue version.
    ///
    /// \param dst_layout New layout after conversion. If not specified - destination layout is obtained from
    /// appropriate tensor output properties.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner.
    PostProcessSteps&& convert_layout(const Layout& dst_layout = {}) &&;

    /// \brief Signature for custom postprocessing operation. Custom postprocessing operation takes one output node and
    /// produces one output node. For more advanced cases, client's code can use transformation passes over ov::Function
    /// directly
    ///
    /// \param node Output node for custom post-processing operation
    ///
    /// \return New node after applying custom post-processing operation
    using CustomPostprocessOp = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>& node)>;

    /// \brief Add custom post-process operation - Lvalue version
    /// Client application can specify callback function for custom action
    ///
    /// \param postprocess_cb Client's custom postprocess operation.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps& custom(const CustomPostprocessOp& postprocess_cb) &;

    /// \brief Add custom post-process operation - Rvalue version
    /// Client application can specify callback function for custom action
    ///
    /// \param postprocess_cb Client's custom postprocess operation.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PostProcessSteps&& custom(const CustomPostprocessOp& postprocess_cb) &&;
};

}  // namespace preprocess
}  // namespace ov
