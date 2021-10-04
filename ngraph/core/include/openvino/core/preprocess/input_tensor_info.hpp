// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/color_format.hpp"
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
    /// \param type Element type for user's input tensor.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo&& set_element_type(const ov::element::Type& type) &&;

    /// \brief Set layout for user's input tensor
    /// This version allows chaining for Lvalue objects
    ///
    /// \param layout Layout for user's input tensor.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo& set_layout(const ov::Layout& layout) &;

    /// \brief Set layout for user's input tensor
    /// This version allows chaining for Rvalue objects
    ///
    /// \param layout Layout for user's input tensor.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo&& set_layout(const ov::Layout& layout) &&;

    /// \brief By default, input image shape is inherited from network input shape. This method specifies that user's
    /// input image has dynamic spatial dimensions (width & height). This can be useful for adding resize preprocessing
    /// from any input image to network's expected dimensions.
    ///
    /// This version allows chaining for Lvalue objects.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_spatial_dynamic_shape() &;

    /// \brief By default, input image shape is inherited from network input shape. This method specifies that user's
    /// input image has dynamic spatial dimensions (width & height). This can be useful for adding resize preprocessing
    /// from any input image to network's expected dimensions.
    ///
    /// This version allows chaining for Rvalue objects.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo&& set_spatial_dynamic_shape() &&;

    /// \brief By default, input image shape is inherited from network input shape. Use this method to specify different
    /// width and height of user's input image. In case if input image size is not known, use
    /// `set_spatial_dynamic_shape` method.
    ///
    /// This version allows chaining for Lvalue objects.
    ///
    /// \param height Set fixed user's input image height.
    ///
    /// \param width Set fixed user's input image width.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo& set_spatial_static_shape(size_t height, size_t width) &;

    /// \brief By default, input image shape is inherited from network input shape. Use this method to specify different
    /// width and height of user's input image. In case if input image size is not known, use
    /// `set_spatial_dynamic_shape` method.
    ///
    /// This version allows chaining for Rvalue objects.
    ///
    /// \param height Set fixed user's input image height.
    ///
    /// \param width Set fixed user's input image width.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner.
    InputTensorInfo&& set_spatial_static_shape(size_t height, size_t width) &&;

    /// \brief Set color format for user's input tensor
    ///
    /// In general way, some formats support multi-plane input, e.g. NV12 image can be represented as 2 separate tensors
    /// (planes): Y plane and UV plane. set_color_format API also allows to set sub_names for such parameters for
    /// convenient usage TBD: example
    ///
    /// This version allows chaining for Lvalue objects
    ///
    /// \param format Color format of input image
    ///
    /// \param sub_names Optional list of sub-names assigned for each plane (e.g. {"Y", "UV"}). If not specified,
    /// sub-names for plane parameters are auto-generated, exact names auto-generation rules depend on specific color
    /// format, and client's code shall not rely on these rules.
    ///
    /// \return Reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo& set_color_format(const ov::preprocess::ColorFormat& format,
                                      const std::vector<std::string>& sub_names = {}) &;

    /// \brief Set color format for user's input tensor
    ///
    /// In general way, some formats support multi-plane input, e.g. NV12 image can be represented as 2 separate tensors
    /// (planes): Y plane and UV plane. set_color_format API also allows to set sub_names for such parameters for
    /// convenient usage TBD: example
    ///
    /// This version allows chaining for Rvalue objects
    ///
    /// \param format Color format of input image
    ///
    /// \param sub_names Optional list of sub-names assigned for each plane (e.g. {"Y", "UV"}). If not specified,
    /// sub-names for plane parameters are auto-generated, exact names auto-generation rules depend on specific color
    /// format, and client's code shall not rely on these rules.
    ///
    /// \return Rvalue reference to 'this' to allow chaining with other calls in a builder-like manner
    InputTensorInfo&& set_color_format(const ov::preprocess::ColorFormat& format,
                                       const std::vector<std::string>& sub_names = {}) &&;
};

}  // namespace preprocess
}  // namespace ov
