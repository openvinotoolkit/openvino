// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_max_poolnd(const NodeContext& context) {
    num_inputs_check(context, 3, 6);
    auto kernel = context.const_input<Shape>(1);
    Strides strides;
    if (!context.input_is_none(2)) {
        strides = context.const_input<Strides>(2);
    }
    if (context.input_is_none(2) || strides.size() == 0) {
        // In case strides are not provided default is kernel
        strides = kernel;
    }
    Shape pads;
    if (context.input_is_none(3)) {
        pads = Shape(kernel.size(), 0);
    } else {
        pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    }
    Strides dilations;
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4);
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR;
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL_TORCH : RoundingType::FLOOR;
    }

    auto res = context.mark_node(std::make_shared<v14::MaxPool>(context.get_input(0),
                                                                strides,
                                                                dilations,
                                                                pads,
                                                                pads,
                                                                kernel,
                                                                rounding_type,
                                                                PadType::EXPLICIT,
                                                                element::i64,
                                                                2));
    if (context.get_output_size() == 2) {
        auto out1 = res->output(0);
        auto out2 = res->output(1);
        return {std::move(out1), std::move(out2)};
    } else {
        return {res};
    }
};

OutputVector translate_max_pool_base(const NodeContext& context, const Output<Node>& input) {
    num_inputs_check(context, 3, 6);
    auto kernel = context.const_input<Shape>(1);
    Strides strides;
    if (!context.input_is_none(2)) {
        strides = context.const_input<Strides>(2);
    }
    if (context.input_is_none(2) || strides.size() == 0) {
        // In case strides are not provided default is kernel
        strides = kernel;
    }
    Shape pads;
    if (context.input_is_none(3)) {
        pads = Shape(kernel.size(), 0);
    } else {
        pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    }
    Strides dilations;
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4);
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR;
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL_TORCH : RoundingType::FLOOR;
    }

    auto res = context.mark_node(std::make_shared<v14::MaxPool>(input,
                                                                strides,
                                                                dilations,
                                                                pads,
                                                                pads,
                                                                kernel,
                                                                rounding_type,
                                                                PadType::EXPLICIT,
                                                                element::i64,
                                                                2));
    if (context.get_output_size() == 2) {
        auto out1 = res->output(0);
        auto out2 = res->output(1);
        return {std::move(out1), std::move(out2)};
    } else {
        return {res};
    }
};

OutputVector translate_max_pool3d(const NodeContext& context) {
    auto input = context.get_input(0);

    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto unsqueeze_axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, unsqueeze_axis));

    // If there was no batch dimension, added dimension by slicing the input tensor
    auto unsqueeze_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(unsqueeze_shape));
    auto end_index = context.mark_node(std::make_shared<v1::Add>(rank,
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}))));

    auto reshape_pattern = context.mark_node(std::make_shared<v8::Slice>(unsqueeze_shape, 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-5})), 
    end_index, 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    input = context.mark_node(std::make_shared<v1::Reshape>(input, reshape_pattern, true));

    auto pooled_output = translate_max_pool_base(context, input);

    // If there was no batch dimension, remove the added dimension by slicing the output tensor
    auto pooled_output_shape = context.mark_node(std::make_shared<v3::ShapeOf>(pooled_output[0]));

    auto slice_input_shape = context.mark_node(std::make_shared<v8::Slice>(input_shape, 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-3})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    auto slice_pooled_output_shape = context.mark_node(std::make_shared<v8::Slice>(pooled_output_shape, 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-3})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {5})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    auto concat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{slice_input_shape, slice_pooled_output_shape}, 0));

    for (auto& node : pooled_output) {
        node = context.mark_node(std::make_shared<v1::Reshape>(node, concat_shape, true));
    }
    return pooled_output; 
};

OutputVector translate_max_pool2d(const NodeContext& context) {
    auto input = context.get_input(0);

    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto unsqueeze_axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, unsqueeze_axis));

    // If there was no batch dimension, added dimension by slicing the input tensor
    auto unsqueeze_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(unsqueeze_shape));
    auto end_index = context.mark_node(std::make_shared<v1::Add>(rank,
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}))));

    auto reshape_pattern = context.mark_node(std::make_shared<v8::Slice>(unsqueeze_shape, 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-4})), 
    end_index, 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    input = context.mark_node(std::make_shared<v1::Reshape>(input, reshape_pattern, true));

    auto pooled_output = translate_max_pool_base(context, input);

    // If there was no batch dimension, remove the added dimension by slicing the output tensor
    auto pooled_output_shape = context.mark_node(std::make_shared<v3::ShapeOf>(pooled_output[0]));

    auto slice_input_shape = context.mark_node(std::make_shared<v8::Slice>(input_shape, 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-2})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    auto slice_pooled_output_shape = context.mark_node(std::make_shared<v8::Slice>(pooled_output_shape, 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-2})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {4})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    auto concat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{slice_input_shape, slice_pooled_output_shape}, 0));

    for (auto& node : pooled_output) {
        node = context.mark_node(std::make_shared<v1::Reshape>(node, concat_shape, true));
    }
    return pooled_output; 
};

OutputVector translate_max_pool1d(const NodeContext& context) {
    auto input = context.get_input(0);

    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto unsqueeze_axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, unsqueeze_axis));

    // If there was no batch dimension, added dimension by slicing the input tensor
    auto unsqueeze_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(unsqueeze_shape));
    auto end_index = context.mark_node(std::make_shared<v1::Add>(rank,
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}))));

    auto reshape_pattern = context.mark_node(std::make_shared<v8::Slice>(unsqueeze_shape, 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-3})), 
    end_index, 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
    context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    input = context.mark_node(std::make_shared<v1::Reshape>(input, reshape_pattern, true));

    auto pooled_output = translate_max_pool_base(context, input);

    // If there was no batch dimension, remove the added dimension by slicing the output tensor
    auto pooled_output_shape = context.mark_node(std::make_shared<v3::ShapeOf>(pooled_output[0]));

    auto slice_input_shape = context.mark_node(std::make_shared<v8::Slice>(input_shape, 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    auto slice_pooled_output_shape = context.mark_node(std::make_shared<v8::Slice>(pooled_output_shape, 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {3})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})), 
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}))));

    auto concat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{slice_input_shape, slice_pooled_output_shape}, 0));

    for (auto& node : pooled_output) {
        node = context.mark_node(std::make_shared<v1::Reshape>(node, concat_shape, true));
    }
    return pooled_output; 
};

OutputVector translate_max_poolnd_fx(const NodeContext& context) {
    auto output = translate_max_poolnd(context);
    return {context.mark_node(make_list_construct(output))};
}

OutputVector translate_max_pool2d_fx(const NodeContext& context) {
    auto output = translate_max_pool2d(context);
    return {context.mark_node(make_list_construct(output))};
}

OutputVector translate_max_pool3d_fx(const NodeContext& context) {
    auto output = translate_max_pool3d(context);
    return {context.mark_node(make_list_construct(output))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
