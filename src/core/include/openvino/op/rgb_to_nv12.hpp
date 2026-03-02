// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/convert_color_to_nv12_base.hpp"

namespace ov {
namespace op {
namespace v16 {
/// \brief Color conversion operation from RGB to NV12 format.
///    Input:
///        - Input RGB image in NHWC layout with shape [N, H, W, 3].
///        - Image dimensions 'H' and 'W' must be even numbers.
///        - Supported element types: u8 or any supported floating-point type.
///    Output:
///        - Output NV12 image can be represented in two ways:
///            a) Single plane (default): Height dimension is 1.5x bigger than image height.
///               Shape: [N, H * 3/2, W, 1].
///               The first H rows contain the Y plane, followed by H/2 rows of interleaved UV.
///            b) Two separate planes:
///               b1) Y plane: [N, H, W, 1]
///               b2) UV plane: [N, H/2, W/2, 2] (interleaved U and V)
///
/// \details Conversion of each pixel from RGB to NV12 (YUV) space is represented by following formulas:
///        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
///        U = -0.148 * R - 0.291 * G + 0.439 * B + 128
///        V = 0.439 * R - 0.368 * G - 0.071 * B + 128
///        Then Y, U, V values are clipped to range (0, 255)
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RGBtoNV12 : public util::ConvertColorToNV12Base {
public:
    OPENVINO_OP("RGBtoNV12", "opset16", util::ConvertColorToNV12Base);

    RGBtoNV12() = default;

    /// \brief Constructs a conversion operation from input image in RGB format.
    /// Output is single-plane NV12.
    ///
    /// \param arg  Node that produces the input tensor (NHWC layout, C=3).
    explicit RGBtoNV12(const Output<Node>& arg);

    /// \brief Constructs a conversion operation from input image in RGB format
    /// with configurable output format.
    ///
    /// \param arg          Node that produces the input tensor (NHWC layout, C=3).
    /// \param single_plane If true, output is single-plane NV12 [N, H*3/2, W, 1].
    ///                     If false, output is two planes: Y [N, H, W, 1] and UV [N, H/2, W/2, 2].
    explicit RGBtoNV12(const Output<Node>& arg, bool single_plane);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v16
}  // namespace op
}  // namespace ov
