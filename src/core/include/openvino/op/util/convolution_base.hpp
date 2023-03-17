// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for operations like convolution, group convolution
class OPENVINO_API ConvolutionBase : public Op {
public:
    OPENVINO_OP("ConvolutionBase", "util");

    /// \brief Constructs a conversion operation.
    ConvolutionBase() = default;

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
    ConvolutionBase(const OutputVector& arguments,
                    const Strides& strides,
                    const CoordinateDiff& pads_begin,
                    const CoordinateDiff& pads_end,
                    const Strides& dilations,
                    const PadType& auto_pad = PadType::EXPLICIT)
        : Op(arguments),
          m_strides(strides),
          m_dilations(dilations),
          m_pads_begin(pads_begin),
          m_pads_end(pads_end),
          m_auto_pad(auto_pad) {}

    const Strides& get_strides() const {
        return m_strides;
    }
    void set_strides(const Strides& strides) {
        m_strides = strides;
    }
    const Strides& get_dilations() const {
        return m_dilations;
    }
    void set_dilations(const Strides& dilations) {
        m_dilations = dilations;
    }
    const CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }
    void set_pads_begin(const CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }
    const CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }
    OPENVINO_DEPRECATED("This method is deprecated and will be removed soon. Please use set_pads_end instead.")
    void set_adding_above(const CoordinateDiff& pads_end) {
        set_pads_end(pads_end);
    }
    void set_pads_end(const CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }
    const PadType& get_auto_pad() const {
        return m_auto_pad;
    }
    void set_auto_pad(const PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }

protected:
    Strides m_strides;
    Strides m_dilations;
    CoordinateDiff m_pads_begin;
    CoordinateDiff m_pads_end;
    PadType m_auto_pad;
    size_t m_num_spatial = std::numeric_limits<size_t>::max();

private:
    friend void resize_attributes(ConvolutionBase* op, const size_t num_spatial);

    template <class TShape>
    friend void apply_padding(ConvolutionBase* op, const TShape& data_shape, const TShape& filters_shape);

    friend bool is_attr_validation_required(const ConvolutionBase* op);
};
}  // namespace util
}  // namespace op
}  // namespace ov
