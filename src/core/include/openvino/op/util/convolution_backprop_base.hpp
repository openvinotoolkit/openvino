// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_base.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for operations like back propagation convolution
class OPENVINO_API ConvolutionBackPropBase : public ConvolutionBase {
public:
    OPENVINO_OP("ConvolutionBackPropBase", "util");

    /// \brief Constructs a conversion operation.
    ConvolutionBackPropBase() = default;

    /// \brief Constructs a conversion operation.
    /// \param strides            Convolution strides.
    /// \param pads_begin         Amount of padding to be added to the beginning along
    ///                           each axis. For example in case of a 2D input the value
    ///                           of (1, 2) means that 1 element will be added to the
    ///                           top and 2 elements to the left.
    /// \param pads_end           Amount of padding to be added to the end along each
    ///                           axis.
    /// \param dilations          The distance in width and height between the weights
    ///                           in the filters tensor.
    /// \param auto_pad           Specifies how the automatic calculation of padding
    ///                           should be done.
    /// \param      output_padding  The output padding adds additional amount of paddings per
    ///                             each spatial axis in the output tensor. clang-format on
    ConvolutionBackPropBase(const OutputVector& arguments,
                            const Strides& strides,
                            const CoordinateDiff& pads_begin,
                            const CoordinateDiff& pads_end,
                            const Strides& dilations,
                            const PadType& auto_pad = PadType::EXPLICIT,
                            const CoordinateDiff& output_padding = {})
        : ConvolutionBase(arguments, strides, pads_begin, pads_end, dilations, auto_pad),
          m_output_padding{output_padding} {}

    const CoordinateDiff& get_output_padding() const {
        return m_output_padding;
    }
    void set_output_padding(const CoordinateDiff& output_padding) {
        m_output_padding = output_padding;
    }

protected:
    CoordinateDiff m_output_padding;

private:
    friend void resize_attributes(ConvolutionBackPropBase* op, const size_t num_spatial);

    template <class TShape>
    friend void apply_padding(ConvolutionBackPropBase* op, const TShape& data_shape, const TShape& filters_shape);

    template <class TShape>
    friend void apply_padding(ConvolutionBackPropBase* op,
                              const TShape& data_shape,
                              const TShape& filters_shape,
                              const TShape& out_spatial_shape);

    friend bool is_attr_validation_required(const ConvolutionBase* op);
};
}  // namespace util
}  // namespace op
}  // namespace ov
