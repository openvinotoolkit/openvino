// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate_diff.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pad(NodeContext& context) {
    auto data = context.get_input(0);
    auto paddings = context.const_input<std::vector<int64_t>>(1);
    std::string mode = "constant";
    double value = 0;
    auto shape = context.mark_node(std::make_shared<opset8::ShapeOf>(data, element::i64));
    auto rank = context.mark_node(std::make_shared<opset8::ShapeOf>(shape, element::i64));
    auto reduced_rank = context.mark_node(std::make_shared<opset8::Squeeze>(rank));
    auto zero = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
    auto pad_begins = context.mark_node(std::make_shared<opset8::Broadcast>(zero, rank));
    auto pad_ends = context.mark_node(std::make_shared<opset8::Broadcast>(zero, rank));
    std::vector<int64_t> pad_b(paddings.size() / 2, 0);
    std::vector<int64_t> pad_e(paddings.size() / 2, 0);
    std::vector<int64_t> neg_indicies(paddings.size() / 2, 0);
    auto pad_mode = ov::op::PadMode::CONSTANT;
    int pad_size = static_cast<int>(paddings.size() / 2);
    for (int i = 0; i < paddings.size() / 2; i++) {
        pad_b.push_back(paddings[paddings.size() - (2 * i + 2)]);
        pad_e.push_back(paddings[paddings.size() - (2 * i + 1)]);
        neg_indicies.push_back(pad_size - i);

    }
    auto neg_id_shape = context.mark_node(opset8::Constant::create(element::i64, Shape({1}), std::vector<size_t>({neg_indicies.size()})));
    auto neg_id_node = context.mark_node(opset8::Constant::create(element::i64, Shape({neg_indicies.size()}), neg_indicies));
    auto end_indicies = context.mark_node(std::make_shared<opset8::Broadcast>(reduced_rank, neg_id_shape));
    auto indicies = context.mark_node(std::make_shared<opset8::Subtract>(end_indicies, neg_id_node));
    pad_begins = context.mark_node(std::make_shared<opset8::ScatterUpdate>(pad_begins, indicies,
            context.mark_node(opset8::Constant::create(element::i64, Shape({pad_b.size()}), pad_b)),
            context.mark_node(opset8::Constant::create(element::i64, Shape{}, {-1}))));
    pad_ends = context.mark_node(std::make_shared<opset8::ScatterUpdate>(pad_ends,
            indicies,
            context.mark_node(opset8::Constant::create(element::i64, Shape({pad_e.size()}), pad_e)),
            context.mark_node(opset8::Constant::create(element::i64, Shape{}, {-1}))));
    if (!context.input_is_none(2)) {
            mode = context.const_input<std::string>(2);
        }
    if (mode == "circular") {
        int64_t pad_l;
        int64_t pad_r;
        auto ndim = paddings.size() / 2;
        auto pad_last_id = paddings.size();
        auto cur = data.get_node_shared_ptr();
        auto step = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));
        for (auto i = 0; i < ndim; i++) {
            ov::NodeVector tensors;
            pad_r = paddings[pad_last_id - (2 * i + 1)];
            pad_l = paddings[pad_last_id - (2 * i + 2)];
            auto axes = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {2 + i}));
            if (pad_l > 0) {
                auto start = context.mark_node(
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1 * pad_l})));
                auto end = context.mark_node(std::make_shared<opset8::Gather>(
                    shape,
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {2 + i})),
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}))));

                auto left = context.mark_node(std::make_shared<opset8::Slice>(cur, start, end, step, axes));
                tensors.push_back(left);
            }
            if (pad_l < 0 || pad_r < 0) {
                auto start = context.mark_node(
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1 * pad_l > 0 ? -1 * pad_l : 0})));
                auto end = context.mark_node(
                    context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1 * (-1 * pad_r > 0 ? -1 * pad_r : 0)})));
                auto middle = context.mark_node(std::make_shared<opset8::Slice>(cur, start, end, step, axes));
                tensors.push_back(middle);
            } else {
                tensors.push_back(cur);
            }
            if (pad_r > 0) {
                auto start = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
                auto end = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {pad_r}));
                auto right = context.mark_node(std::make_shared<opset8::Slice>(cur, start, end, step, axes));
                tensors.push_back(right);
            }
            if (tensors.size()){
            cur = context.mark_node(std::make_shared<opset8::Concat>(tensors, 2 + i));
            }
        }
        return {cur};
    }
    if (mode == "constant") {
        pad_mode = ov::op::PadMode::CONSTANT;
        if (!context.input_is_none(3)) {
            value = context.const_input<double>(3);
        }
    } else if (mode == "reflect") {
        pad_mode = ov::op::PadMode::REFLECT;
    } else if (mode == "replicate") {
        pad_mode = ov::op::PadMode::EDGE;
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "aten::pad conversion doesn't support [" + mode + "] padding mode");
    }
    return {context.mark_node(std::make_shared<opset8::Pad>(
        data,
        pad_begins,
        pad_ends,
        context.mark_node(std::make_shared<opset8::Constant>(element::f32, Shape({}), value)),
        pad_mode))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov