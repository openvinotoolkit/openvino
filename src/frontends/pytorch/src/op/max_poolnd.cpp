// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
OutputVector translate_max_pool_base(const NodeContext& context, int dims, bool return_indices) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto const_0 = v0::Constant::create(element::i64, Shape{1}, {0});
    auto const_1 = v0::Constant::create(element::i64, Shape{1}, {1});
    bool is_static = input.get_partial_shape().rank().is_static();
    bool no_batch_dim = is_static && input.get_partial_shape().rank().get_length() == dims + 1;

    if (is_static) {
        if (no_batch_dim) {
            input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_0));
        }
    } else {
        input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_0));
        auto unsqueeze_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
        auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(unsqueeze_shape));
        auto end_index = context.mark_node(std::make_shared<v1::Add>(rank, const_1));
        auto start_index = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-dims - 2}));
        auto reshape_pattern =
            context.mark_node(std::make_shared<v8::Slice>(unsqueeze_shape, start_index, end_index, const_1, const_0));
        input = context.mark_node(std::make_shared<v1::Reshape>(input, reshape_pattern, true));
    }

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
    auto dilations = Strides(dims, 1);
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
    if (is_static) {
        if (no_batch_dim) {
            if (return_indices) {
                auto out1 = res->output(0);
                auto out2 = res->output(1);
                out1 = context.mark_node(std::make_shared<v0::Squeeze>(out1, const_0));
                out2 = context.mark_node(std::make_shared<v0::Squeeze>(out2, const_0));
                return {std::move(out1), std::move(out2)};
            } else {
                res = context.mark_node(std::make_shared<v0::Squeeze>(res, const_0));
                return {res};
            }
        } else {
            if (return_indices) {
                auto out1 = res->output(0);
                auto out2 = res->output(1);
                return {std::move(out1), std::move(out2)};
            } else {
                return {res};
            }
        }

    } else {
        auto pooled_output_shape = context.mark_node(std::make_shared<v3::ShapeOf>(res));

        auto start_index_input = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-dims}));
        auto slice_input_shape =
            context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0, start_index_input, const_1, const_0));

        auto start_index_pooled = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-dims}));
        auto end_index_pooled = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {2 + dims}));
        auto slice_pooled_output_shape = context.mark_node(
            std::make_shared<v8::Slice>(pooled_output_shape, start_index_pooled, end_index_pooled, const_1, const_0));

        auto concat_shape = context.mark_node(
            std::make_shared<v0::Concat>(OutputVector{slice_input_shape, slice_pooled_output_shape}, 0));
        if (return_indices) {
            auto out1 = res->output(0);
            auto out2 = res->output(1);
            out1 = context.mark_node(std::make_shared<v1::Reshape>(out1, concat_shape, true));
            out2 = context.mark_node(std::make_shared<v1::Reshape>(out2, concat_shape, true));
            return {std::move(out1), std::move(out2)};
        } else {
            res = context.mark_node(std::make_shared<v1::Reshape>(res, concat_shape, true));
            return {res};
        }
    }
};

OutputVector translate_max_pool1d(const NodeContext& context) {
    return translate_max_pool_base(context, 1, context.get_output_size() == 2);
};

OutputVector translate_max_pool2d(const NodeContext& context) {
    return translate_max_pool_base(context, 2, context.get_output_size() == 2);
};

OutputVector translate_max_pool3d(const NodeContext& context) {
    return translate_max_pool_base(context, 3, context.get_output_size() == 2);
};

OutputVector translate_max_pool2d_fx(const NodeContext& context) {
    auto output = translate_max_pool_base(context, 2, true);
    return {context.mark_node(make_list_construct(output))};
};

OutputVector translate_max_pool3d_fx(const NodeContext& context) {
    auto output = translate_max_pool_base(context, 3, true);
    return {context.mark_node(make_list_construct(output))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
