// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
ov::OutputVector translate_interpolate_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ResizeBilinear", "ResizeNearestNeighbor"});
    auto images = node.get_input(0);
    auto size = node.get_input(1);
    auto op_name = node.get_name();
    auto op_type = node.get_op_type();

    // retrieve optional attribute
    auto tf_align_corners = node.get_attribute<bool>("align_corners", false);
    auto tf_half_pixel_centers = node.get_attribute<bool>("half_pixel_centers", false);

    TENSORFLOW_OP_VALIDATION(node,
                             !tf_half_pixel_centers || (tf_half_pixel_centers && !tf_align_corners),
                             "If half_pixel_centers attribute of the node" + op_name + " with op " + op_type +
                                 " is True, the attribute align_corners must be False.");

    // prepare attributes for OpenVINO Interpolate operation
    Interpolate::InterpolateAttrs interpolate_attrs;
    interpolate_attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
    if (op_type == "ResizeNearestNeighbor") {
        interpolate_attrs.mode = Interpolate::InterpolateMode::NEAREST;
        interpolate_attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
    } else if (op_type == "ResizeBilinear") {
        interpolate_attrs.mode = Interpolate::InterpolateMode::LINEAR;
        interpolate_attrs.nearest_mode = Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    }

    if (tf_align_corners) {
        interpolate_attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        if (interpolate_attrs.mode == Interpolate::InterpolateMode::NEAREST) {
            interpolate_attrs.nearest_mode = Interpolate::NearestMode::ROUND_PREFER_CEIL;
        }
    } else if (tf_half_pixel_centers) {
        if (interpolate_attrs.mode == Interpolate::InterpolateMode::NEAREST) {
            interpolate_attrs.coordinate_transformation_mode =
                Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN;
        } else {
            interpolate_attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::HALF_PIXEL;
        }
    } else {
        interpolate_attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
    }

    // prepare scales input
    auto images_shape = make_shared<ShapeOf>(images, ov::element::i32);
    auto spatial_shape = make_shared<Slice>(images_shape,
                                            make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                            make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{3}),
                                            make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                            make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>{0}));
    auto scales = make_shared<Divide>(make_shared<Convert>(size, element::f32),
                                      make_shared<Convert>(spatial_shape, element::f32));

    // since Interpolate is layout agnostic
    // we can avoid Transpose operation by specifying axes = {1, 2} for original NHWC layout
    auto axes = make_shared<Constant>(element::i32, Shape{2}, std::vector<int>({1, 2}));

    auto interpolate = make_shared<Interpolate>(images, size, scales, axes, interpolate_attrs);
    set_node_name(node.get_name(), interpolate);
    return {interpolate};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
