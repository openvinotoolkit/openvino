// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Base class for color conversion operation from RGB/BGR to NV12 format.
///    Input:
///        - Operation expects input shape in NHWC layout.
///        - Input RGB/BGR image has shape [N, H, W, 3]. H and W must be even.
///        - Supported element types: u8 or any supported floating-point type.
///    Output:
///        - Output NV12 image can be represented in two ways:
///            a) Single plane: Height dimension is 1.5x bigger than image height. 'C' dimension equals 1.
///               Shape: [N, H * 3/2, W, 1]
///            b) Two separate planes:
///               b1) Y plane has height same as image height. 'C' dimension equals 1.
///                   Shape: [N, H, W, 1]
///               b2) UV plane has dimensions: 'H' = image_h / 2; 'W' = image_w / 2; 'C' = 2.
///                   Shape: [N, H/2, W/2, 2]
///
/// \details Conversion of each pixel from RGB to NV12 (YUV) space is represented by following formulas:
///        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
///        U = -0.148 * R - 0.291 * G + 0.439 * B + 128
///        V = 0.439 * R - 0.368 * G - 0.071 * B + 128
///        Then Y, U, V values are clipped to range (0, 255)
///        UV plane is subsampled by averaging each 2x2 block of pixels.
///
class OPENVINO_API ConvertColorToNV12Base : public Op {
public:
    /// \brief Exact conversion format details
    enum class ColorConversion : int { RGB_TO_NV12 = 0, BGR_TO_NV12 = 1 };

protected:
    ConvertColorToNV12Base() = default;

    /// \brief Constructs a conversion operation from input image in RGB/BGR format.
    /// Default output is single-plane NV12.
    ///
    /// \param arg          Node that produces the input tensor. Input tensor represents image in RGB/BGR format (NHWC).
    /// \param format       Conversion format.
    explicit ConvertColorToNV12Base(const Output<Node>& arg, ColorConversion format);

    /// \brief Constructs a conversion operation from input image in RGB/BGR format with configurable output.
    ///
    /// \param arg          Node that produces the input tensor. Input tensor represents image in RGB/BGR format (NHWC).
    /// \param format       Conversion format.
    /// \param single_plane If true, output is single-plane NV12; if false, output is Y and UV as separate planes.
    ConvertColorToNV12Base(const Output<Node>& arg, ColorConversion format, bool single_plane);

public:
    OPENVINO_OP("ConvertColorToRGBBase", "util");

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool is_single_plane() const {
        return m_single_plane;
    }

protected:
    bool is_type_supported(const ov::element::Type& type) const;

    ColorConversion m_format = ColorConversion::RGB_TO_NV12;
    bool m_single_plane = true;
};
}  // namespace util
}  // namespace op
}  // namespace ov
