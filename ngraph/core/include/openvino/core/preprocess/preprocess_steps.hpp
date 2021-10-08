// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/preprocess/color_format.hpp"
#include "openvino/core/preprocess/resize_algorithm.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

class Node;

namespace preprocess {

/// \brief Preprocessing steps. Each step typically intends adding of some operation to input parameter
/// User application can specify sequence of preprocessing steps in a builder-like manner
/// \code{.cpp}
/// auto proc = PrePostProcessor()
///     .input(InputInfo()
///            .preprocess(PreProcessSteps()
///                        .mean(0.2f)     // Subtract 0.2 from each element
///                        .scale(2.3f))   // then divide each element to 2.3
///     );
/// \endcode
class OPENVINO_API PreProcessSteps final {
    class PreProcessStepsImpl;
    std::unique_ptr<PreProcessStepsImpl> m_impl;
    friend class InputInfo;

public:
    /// \brief Default empty constructor
    PreProcessSteps();

    /// \brief Default move constructor
    PreProcessSteps(PreProcessSteps&&) noexcept;

    /// \brief Default move assignment operator
    PreProcessSteps& operator=(PreProcessSteps&&) noexcept;

    /// \brief Default destructor
    ~PreProcessSteps();

    /// \brief Add convert element type preprocess operation - Lvalue version
    ///
    /// \param type Desired type of input.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& convert_element_type(const ov::element::Type& type = {}) &;

    /// \brief Add convert element type preprocess operation - Rvalue version
    ///
    /// \param type Desired type of input.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& convert_element_type(const ov::element::Type& type = {}) &&;

    /// \brief Converts color format for user's input tensor. Requires source color format to be specified by
    /// InputTensorInfo::set_color_format.
    ///
    /// This version allows chaining for Lvalue objects
    ///
    /// \param dst_format Destination color format of input image
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& convert_color(const ov::preprocess::ColorFormat& dst_format) &;

    /// \brief Converts color format for user's input tensor. Requires source color format to be specified by
    /// InputTensorInfo::set_color_format.
    ///
    /// This version allows chaining for Rvalue objects.
    ///
    /// \param dst_format Color format of input image.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& convert_color(const ov::preprocess::ColorFormat& dst_format) &&;

    /// \brief Add scale preprocess operation - Lvalue version
    /// Divide each element of input by specified value
    ///
    /// \param value Scaling value.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& scale(float value) &;

    /// \brief Add scale preprocess operation - Rvalue version
    /// Divide each element of input by specified value
    ///
    /// \param value Scaling value.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& scale(float value) &&;

    /// \brief Add scale preprocess operation - Lvalue version
    ///
    /// \param values Scaling values. Layout runtime info with channels dimension must be specified for input tensor
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& scale(const std::vector<float>& values) &;

    /// \brief Add scale preprocess operation - Rvalue version
    ///
    /// \param values Scaling values. Layout runtime info with channels dimension must be specified for input tensor
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& scale(const std::vector<float>& values) &&;

    /// \brief Add mean preprocess operation - Lvalue version
    /// Subtract specified value from each element of input
    ///
    /// \param value Value to subtract from each element.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& mean(float value) &;

    /// \brief Add mean preprocess operation - Rvalue version
    /// Subtract specified value from each element of input
    ///
    /// \param value Value to subtract from each element.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& mean(float value) &&;

    /// \brief Add mean preprocess operation - Lvalue version
    ///
    /// \param values Mean values. Layout runtime info with channels dimension must be specified for input tensor
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& mean(const std::vector<float>& values) &;

    /// \brief Add mean preprocess operation - Rvalue version
    ///
    /// \param values Mean values. Layout runtime info with channels dimension must be specified for input tensor
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& mean(const std::vector<float>& values) &&;

    /// \brief Signature for custom preprocessing operation. Custom preprocessing operation takes one input node and
    /// produces one output node. For more advanced cases, client's code can use transformation passes over ov::Function
    /// directly
    ///
    /// \param node Input node for custom preprocessing operation (output of previous preprocessing operation)
    ///
    /// \return New node after applying custom preprocessing operation
    using CustomPreprocessOp = std::function<Output<Node>(const Output<Node>& node)>;

    /// \brief Add custom preprocess operation - Lvalue version
    /// Client application can specify callback function for custom action
    ///
    /// \param preprocess_cb Client's custom preprocess operation.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps& custom(const CustomPreprocessOp& preprocess_cb) &;

    /// \brief Add custom preprocess operation - Rvalue version
    /// Client application can specify callback function for custom action
    ///
    /// \param preprocess_cb Client's custom preprocess operation.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    PreProcessSteps&& custom(const CustomPreprocessOp& preprocess_cb) &&;

    /// \brief Add resize operation to known dimensions - Lvalue version.
    ///
    /// \param alg Resize algorithm.
    ///
    /// \param dst_height Desired height of resized image.
    ///
    /// \param dst_width Desired width of resized image.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    PreProcessSteps& resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) &;

    /// \brief Add resize operation to known dimensions - Rvalue version.
    ///
    /// \param alg Resize algorithm.
    ///
    /// \param dst_height Desired height of resized image.
    ///
    /// \param dst_width Desired width of resized image.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner.
    PreProcessSteps&& resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) &&;

    /// \brief Add resize operation to network dimensions - Lvalue version.
    ///
    /// \param alg Resize algorithm.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    PreProcessSteps& resize(ResizeAlgorithm alg) &;

    /// \brief Add resize operation to network dimensions - Rvalue version.
    ///
    /// \param alg Resize algorithm.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner.
    PreProcessSteps&& resize(ResizeAlgorithm alg) &&;

    /// \brief Add 'convert layout' operation to specified layout - Lvalue version.
    ///
    /// \details Adds appropriate 'transpose' operation between user layout and target layout.
    /// Current implementation requires source and destination layout to have same number of dimensions
    ///
    /// \example Example: when user data has 'NHWC' layout (example is RGB image, [1, 224, 224, 3]) but network expects
    /// planar input image ('NCHW', [1, 3, 224, 224]). Preprocessing may look like this:
    ///
    /// \code{.cpp} auto proc =
    /// PrePostProcessor()
    ///     .input(InputInfo()
    ///            .tensor(InputTensorInfo().set_layout("NHWC")) // User data is NHWC
    ///            .preprocess(PreProcessSteps()
    ///                        .convert_layout("NCHW")) // Network expects input as NCHW
    ///     );
    /// \endcode
    ///
    /// \param dst_layout New layout after conversion. If not specified - destination layout is obtained from
    /// appropriate network input properties.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    PreProcessSteps& convert_layout(const Layout& dst_layout = {}) &;

    /// \brief Add resize operation to network dimensions - Rvalue version.
    ///
    /// \param dst_layout New layout after conversion. If not specified - destination layout is obtained from
    /// appropriate network input properties.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner.
    PreProcessSteps&& convert_layout(const Layout& dst_layout = {}) &&;
};

}  // namespace preprocess
}  // namespace ov
