// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> input_or_const(const NodeContext& context, size_t idx, const Output<Node>& default_value) {
    if (context.get_input_size() > idx && !context.input_is_none(idx)) {
        return context.get_input(idx);
    }
    return default_value;
}
}  // namespace

OutputVector translate_histc(const NodeContext& context) {
    // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
    // aten::histc.out(..., Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 5);
    auto input = context.get_input(0);
    const auto orig_type = input.get_element_type();

    if (orig_type.is_static() && !orig_type.is_real()) {
        input = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
    }

    auto bins = input_or_const(context, 1, context.mark_node(v0::Constant::create(element::i64, Shape{}, {100})));
    auto min_val = input_or_const(context, 2, context.mark_node(v0::Constant::create(element::f32, Shape{}, {0})));
    auto max_val = input_or_const(context, 3, context.mark_node(v0::Constant::create(element::f32, Shape{}, {0})));

    auto minus_one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto flat = context.mark_node(std::make_shared<v1::Reshape>(input, minus_one, false));
    min_val = context.mark_node(std::make_shared<v1::ConvertLike>(min_val, flat));
    max_val = context.mark_node(std::make_shared<v1::ConvertLike>(max_val, flat));

    auto reduce_axes = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto data_min = context.mark_node(std::make_shared<v1::ReduceMin>(flat, reduce_axes, false));
    auto data_max = context.mark_node(std::make_shared<v1::ReduceMax>(flat, reduce_axes, false));
    auto numel = context.mark_node(
        std::make_shared<v1::ReduceProd>(context.mark_node(std::make_shared<v3::ShapeOf>(flat, element::i64)),
                                         reduce_axes,
                                         false));
    auto has_data = context.mark_node(
        std::make_shared<v1::Greater>(numel, context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}))));
    auto min_eq_max = context.mark_node(std::make_shared<v1::Equal>(min_val, max_val));
    auto infer_range = context.mark_node(std::make_shared<v1::LogicalAnd>(min_eq_max, has_data));
    auto left = context.mark_node(std::make_shared<v1::Select>(infer_range, data_min, min_val));
    auto right = context.mark_node(std::make_shared<v1::Select>(infer_range, data_max, max_val));

    // When the range is still empty (constant input or empty tensor), expand by ±1 like PyTorch histc.
    auto still_eq = context.mark_node(std::make_shared<v1::Equal>(left, right));
    auto one = context.mark_node(
        std::make_shared<v1::ConvertLike>(context.mark_node(v0::Constant::create(element::f32, Shape{}, {1})), left));
    left = context.mark_node(
        std::make_shared<v1::Select>(still_eq, context.mark_node(std::make_shared<v1::Subtract>(left, one)), left));
    right = context.mark_node(
        std::make_shared<v1::Select>(still_eq, context.mark_node(std::make_shared<v1::Add>(right, one)), right));

    auto bins_f = context.mark_node(std::make_shared<v1::ConvertLike>(bins, flat));
    auto width = context.mark_node(
        std::make_shared<v1::Divide>(context.mark_node(std::make_shared<v1::Subtract>(right, left)), bins_f));
    auto pos = context.mark_node(std::make_shared<v0::Floor>(context.mark_node(
        std::make_shared<v1::Divide>(context.mark_node(std::make_shared<v1::Subtract>(flat, left)), width))));

    auto zero_f = context.mark_node(
        std::make_shared<v1::ConvertLike>(context.mark_node(v0::Constant::create(element::f32, Shape{}, {0})), flat));
    auto last_bin_f = context.mark_node(
        std::make_shared<v1::ConvertLike>(context.mark_node(std::make_shared<v1::Subtract>(
                                              context.mark_node(std::make_shared<v0::Convert>(bins, element::i64)),
                                              context.mark_node(v0::Constant::create(element::i64, Shape{}, {1})))),
                                          flat));
    auto clamped = context.mark_node(
        std::make_shared<v1::Minimum>(context.mark_node(std::make_shared<v1::Maximum>(pos, zero_f)), last_bin_f));

    auto in_range = context.mark_node(
        std::make_shared<v1::LogicalAnd>(context.mark_node(std::make_shared<v1::GreaterEqual>(flat, left)),
                                         context.mark_node(std::make_shared<v1::LessEqual>(flat, right))));
    auto safe_pos = context.mark_node(std::make_shared<v1::Select>(in_range, clamped, zero_f));
    auto bin_idx = context.mark_node(std::make_shared<v0::Convert>(safe_pos, element::i64));

    auto flat_shape = context.mark_node(std::make_shared<v3::ShapeOf>(flat, element::i64));
    auto ones = context.mark_node(
        std::make_shared<v3::Broadcast>(context.mark_node(std::make_shared<v1::ConvertLike>(
                                            context.mark_node(v0::Constant::create(element::f32, Shape{}, {1})),
                                            flat)),
                                        flat_shape));
    auto updates = context.mark_node(std::make_shared<v1::Select>(in_range, ones, zero_f));

    auto hist_shape = context.mark_node(
        std::make_shared<v1::Reshape>(context.mark_node(std::make_shared<v0::Convert>(bins, element::i64)),
                                      context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1})),
                                      false));
    auto histogram = context.mark_node(std::make_shared<v3::Broadcast>(zero_f, hist_shape));
    auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto result =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(histogram,
                                                                       bin_idx,
                                                                       updates,
                                                                       axis,
                                                                       v12::ScatterElementsUpdate::Reduction::SUM));

    if (orig_type.is_static() && orig_type.is_real() && orig_type != result->get_element_type()) {
        result = context.mark_node(std::make_shared<v0::Convert>(result, orig_type));
    }

    if (context.get_input_size() > 4 && !context.input_is_none(4)) {
        context.mutate_input(4, result);
    }
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
