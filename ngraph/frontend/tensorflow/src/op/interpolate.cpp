// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
ngraph::OutputVector TranslateInterpolateOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto input_sizes = node.get_ng_input(1);

    Interpolate::InterpolateAttrs interpolate_attrs;
    interpolate_attrs.mode = Interpolate::InterpolateMode::LINEAR;
    interpolate_attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
    if (node.get_attribute<bool>("align_corners", false))
        interpolate_attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ALIGN_CORNERS;

    if (node.get_op_type() == "ResizeNearestNeighbor") {
        interpolate_attrs.mode = Interpolate::InterpolateMode::NEAREST;
        interpolate_attrs.nearest_mode = Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    }

    // todo (itikhono): do we need this .get_shape() actually?
    auto input_shape = input.get_shape();
    std::vector<float> spatial_shape = {static_cast<float>(input_shape[1]), static_cast<float>(input_shape[2])};
    auto ng_spatial_shape = make_shared<Constant>(element::f32, Shape{2}, spatial_shape);

    auto ng_sizes = make_shared<Convert>(input_sizes, element::f32);
    auto ng_scales = make_shared<Divide>(ng_sizes, ng_spatial_shape);
    auto ng_axes = make_shared<Constant>(element::i32, Shape{2}, std::vector<int>({2, 3}));

    Transpose<0, 3, 1, 2>(input);
    auto interpolate = make_shared<Interpolate>(input, input_sizes, ng_scales, ng_axes, interpolate_attrs)->output(0);
    Transpose<0, 2, 3, 1>(interpolate);
    interpolate.get_node_shared_ptr()->set_friendly_name(node.get_name());
    return {interpolate};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
