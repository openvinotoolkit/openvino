// Copyright (C) 2018-2022 Intel Corporation
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

OutputVector translate_crop_and_resize_op(const NodeContext& node) {
    /// ng_input: [batch, image_height, image_width, depth]
    /// ng_boxes: [num_boxes, 4]; each box is a normalized [0.to 1.] co-ordinate
    /// [y1,
    /// x1, y2, x2]
    /// ng_box_ind: [num_boxes]; i-th ng_box_ind refers to the image to crop and
    /// ranges from 0 to batch
    /// ng_crop_size: [crop_height, crop_width];

    /// for each box b specified in ng_boxes:
    ///  1. crop ng_input[ng_box_ind[b]] w/ co-ordinates in ng_boxes
    ///  2. resize according to method

    auto ng_input = node.get_input(0);
    auto ng_boxes = node.get_input(1);
    auto ng_box_ind = node.get_input(2);
    auto ng_size = node.get_input(3);

    auto resize_method = node.get_attribute<string>("method");

    TENSORFLOW_OP_VALIDATION(node,
                             ng_input.get_partial_shape().is_static() && ng_boxes.get_partial_shape().is_static() &&
                                 ng_box_ind.get_partial_shape().is_static() && ng_size.get_partial_shape().is_static(),
                             "Dynamic shapes are not supported.");

    auto spatial_shape = ng_input.get_shape();
    auto image_height = spatial_shape[1];
    auto image_width = spatial_shape[2];
    auto image_depth = spatial_shape[3];

    auto const_boxes = dynamic_pointer_cast<Constant>(ng_boxes.get_node_shared_ptr());
    auto const_box_ind = dynamic_pointer_cast<Constant>(ng_box_ind.get_node_shared_ptr());
    auto const_crop_size = dynamic_pointer_cast<Constant>(ng_size.get_node_shared_ptr());

    TENSORFLOW_OP_VALIDATION(node,
                             const_boxes && const_box_ind && const_crop_size,
                             "Boxes, BoxIndexes, CropSize inputs must be constant.");

    auto boxes = const_boxes->cast_vector<float>();
    auto box_ind = const_box_ind->cast_vector<int64_t>();
    auto crop_size = const_crop_size->cast_vector<int64_t>();

    OutputVector ng_crop_outputs(box_ind.size());
    if (box_ind.empty()) {
        return make_shared<Constant>(element::f32,
                                     Shape{0,
                                           static_cast<unsigned long>(crop_size.at(0)),
                                           static_cast<unsigned long>(crop_size.at(1)),
                                           image_depth},
                                     vector<float>({}))
            ->outputs();
    } else {
        for (int i = 0; i < box_ind.size(); i++) {
            int y1, x1, y2, x2;
            y1 = static_cast<int>(boxes.at(0 + i * 4) * (image_height - 1.));
            x1 = static_cast<int>(boxes.at(1 + i * 4) * (image_width - 1.));
            y2 = static_cast<int>(boxes.at(2 + i * 4) * (image_height - 1.));
            x2 = static_cast<int>(boxes.at(3 + i * 4) * (image_width - 1.));

            int crop_height = abs(y2 - y1);
            int crop_width = abs(x2 - x1);

            // account for flip crops when y1>y2 or x1>x2 with negative striding
            int stride_height = 1, stride_width = 1;
            if (y1 > y2) {
                y1 = y1 - static_cast<int>(image_height);
                y2 = y2 - static_cast<int>(image_height) - 2;
                stride_height = -1;
            }
            if (x1 > x2) {
                x1 = x1 - static_cast<int>(image_height);
                x2 = x2 - static_cast<int>(image_height) - 2;
                stride_width = -1;
            }

            auto begin = make_shared<Constant>(element::i64,
                                               Shape{4},
                                               vector<int64_t>({static_cast<int64_t>(box_ind[i]), y1, x1, 0}));
            auto end = make_shared<Constant>(
                element::i64,
                Shape{4},
                vector<int64_t>(
                    {static_cast<int64_t>(box_ind[i]) + 1, y2 + 1, x2 + 1, static_cast<int64_t>(image_depth + 1)}));
            auto strides =
                make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>({1, stride_height, stride_width, 1}));

            // crop
            auto ng_crop_node =
                make_shared<StridedSlice>(ng_input, begin, end, strides, vector<int64_t>{}, vector<int64_t>{});
            auto ng_crop = ng_crop_node->output(0);

            Interpolate::InterpolateAttrs interpolate_attrs;
            // always corner aligned
            interpolate_attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ALIGN_CORNERS;

            // TODO: handle the case when extrapolation value is greatger than 1.0
            // arguments for resizing
            auto ng_spatial_shape =
                make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{crop_height, crop_width});
            auto ng_input_shape = make_shared<Convert>(ng_spatial_shape, element::f32);
            auto ng_crop_size = make_shared<Convert>(ng_size, element::f32);
            auto ng_scales = make_shared<Divide>(ng_crop_size, ng_input_shape);
            auto ng_axes = make_shared<Constant>(element::i32, Shape{2}, vector<int>({2, 3}));

            if (resize_method == "bilinear") {
                interpolate_attrs.mode = Interpolate::InterpolateMode::LINEAR;
            } else {  // nearest
                interpolate_attrs.mode = Interpolate::InterpolateMode::NEAREST;
            }

            ng_crop = make_transpose(ng_crop, {0, 3, 1, 2});
            auto ng_output =
                make_shared<Interpolate>(ng_crop, ng_size, ng_scales, ng_axes, interpolate_attrs)->output(0);
            ng_output = make_transpose(ng_output, {0, 2, 3, 1});
            ng_crop_outputs.at(i) = ng_output;
        }

        auto res = make_shared<Concat>(ng_crop_outputs, 0);
        set_node_name(node.get_name(), res);
        return res->outputs();
    }
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov