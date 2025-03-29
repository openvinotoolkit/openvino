// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_interpolate_op(const NodeContext& node) {
    default_op_checks(node,
                      2,
                      {"ResizeBilinear", "ResizeNearestNeighbor", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"});
    auto images = node.get_input(0);
    auto size = node.get_input(1);
    auto op_name = node.get_name();
    auto op_type = node.get_op_type();
    auto input_type = images.get_element_type();

    // retrieve optional attribute
    auto tf_align_corners = node.get_attribute<bool>("align_corners", false);
    auto tf_half_pixel_centers = node.get_attribute<bool>("half_pixel_centers", false);

    TENSORFLOW_OP_VALIDATION(node,
                             !tf_half_pixel_centers || (tf_half_pixel_centers && !tf_align_corners),
                             "If half_pixel_centers attribute of the node" + op_name + " with op " + op_type +
                                 " is True, the attribute align_corners must be False.");

    // prepare attributes for OpenVINO Interpolate operation
    v11::Interpolate::InterpolateAttrs interpolate_attrs;
    interpolate_attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SIZES;
    if (op_type == "ResizeNearestNeighbor" || op_type == "RESIZE_NEAREST_NEIGHBOR") {
        interpolate_attrs.mode = v11::Interpolate::InterpolateMode::NEAREST;
        interpolate_attrs.nearest_mode = v11::Interpolate::NearestMode::FLOOR;
    } else if (op_type == "ResizeBilinear" || op_type == "RESIZE_BILINEAR") {
        auto input_rank = images.get_partial_shape().rank();
        if (input_rank.is_static() && input_rank.get_length() == 4) {
            interpolate_attrs.mode = v11::Interpolate::InterpolateMode::LINEAR_ONNX;
        } else {
            interpolate_attrs.mode = v11::Interpolate::InterpolateMode::LINEAR;
        }
        interpolate_attrs.nearest_mode = v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    }

    if (tf_align_corners) {
        interpolate_attrs.coordinate_transformation_mode = v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        if (interpolate_attrs.mode == v11::Interpolate::InterpolateMode::NEAREST) {
            interpolate_attrs.nearest_mode = v11::Interpolate::NearestMode::ROUND_PREFER_CEIL;
        }
    } else if (tf_half_pixel_centers) {
        if (interpolate_attrs.mode == v11::Interpolate::InterpolateMode::NEAREST) {
            interpolate_attrs.coordinate_transformation_mode =
                v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN;
        } else {
            interpolate_attrs.coordinate_transformation_mode = v11::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        }
    } else {
        interpolate_attrs.coordinate_transformation_mode = v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    }

    // since Interpolate is layout agnostic
    // we can avoid Transpose operation by specifying axes = {1, 2} for original NHWC layout
    auto axes = make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int>({1, 2}));

    // according to the specification of ResizeBilinear,
    // it always returns FP32 output type so we immediately align input type for it
    if (op_type == "ResizeBilinear" || op_type == "RESIZE_BILINEAR") {
        images = make_shared<v0::Convert>(images, element::f32);
    } else if (input_type == element::i16 || input_type == element::u16) {
        // OV Interpolate does not support i16 and u16 so it needs to adjust it temporarily
        // OV interpolate supports only f32, f16, bf16, i8, u8, i64, i32
        images = make_shared<v0::Convert>(images, element::i32);
    }

    Output<Node> interpolate = make_shared<v11::Interpolate>(images, size, axes, interpolate_attrs);

    // it needs to return original i16 and u16 input_type due to temporal adjustment before
    // this is not required for ResizeBilinear case since it always returns f32 by specification
    if (op_type != "ResizeBilinear" && op_type != "RESIZE_BILINEAR" &&
        (input_type == element::i16 || input_type == element::u16)) {
        interpolate = make_shared<v0::Convert>(interpolate, input_type);
    }

    set_node_name(node.get_name(), interpolate.get_node_shared_ptr());
    return {interpolate};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
