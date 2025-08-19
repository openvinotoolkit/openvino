// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/attribute.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

namespace detail {
static std::shared_ptr<ov::Node> get_axes_range(const ov::Output<ov::Node>& input) {
    const auto shape_of_input = std::make_shared<v3::ShapeOf>(input);
    const auto scalar = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    const auto rank_of_input = std::make_shared<v3::ShapeOf>(shape_of_input);
    const auto rank_of_input_scalar = std::make_shared<v0::Squeeze>(rank_of_input, scalar);
    const auto start = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto step = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    return std::make_shared<v4::Range>(start, rank_of_input_scalar, step, ov::element::i64);
}

ov::OutputVector center_crop_pad_impl(const ov::OutputVector inputs, const std::vector<int64_t>& axes_attr) {
    const auto& data = inputs[0];
    const auto& target = inputs[1];
    const auto target_i64 = std::make_shared<v0::Convert>(target, ov::element::i64);

    ov::Output<ov::Node> axes_output;
    if (axes_attr.empty()) {
        const auto target_shape_of = std::make_shared<v3::ShapeOf>(target, ov::element::i64);
        const auto target_len_scalar = std::make_shared<v0::Squeeze>(target_shape_of);
        const auto start = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto step = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        axes_output = std::make_shared<v4::Range>(start, target_len_scalar, step, ov::element::i64);
    } else {
        std::vector<int64_t> pos = axes_attr;
        if (data.get_partial_shape().rank().is_static()) {
            int64_t r = data.get_partial_shape().rank().get_length();
            for (auto& a : pos) 
                if (a < 0) 
                    a += r;
        }
        const auto axes_const = std::make_shared<v0::Constant>(ov::element::i64, Shape{pos.size()}, pos);
        axes_output = axes_const;
    }

    const auto in_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    const auto axis_const = std::make_shared<v0::Constant>(ov::element::i64, Shape{}, std::vector<int64_t>{0});
    const auto full_target = std::make_shared<v3::ScatterElementsUpdate>(in_shape, axes_output, target_i64, axis_const);

    const auto desired = std::make_shared<v1::Minimum>(in_shape, full_target);
    const auto diff_crop = std::make_shared<v1::Subtract>(in_shape, desired);
    const auto two_const = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{2});
    const auto half_crop = std::make_shared<v1::Divide>(diff_crop, two_const);

    const auto ends = std::make_shared<v1::Add>(half_crop, desired);
    const auto one_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto rank_1d = std::make_shared<v3::ShapeOf>(in_shape, ov::element::i64);
    const auto step_tensor = std::make_shared<v3::Broadcast>(one_scalar, rank_1d);
    const auto all_axes = get_axes_range(data);
    const auto cropped = std::make_shared<v8::Slice>(data, half_crop, ends, step_tensor, all_axes);

    const auto diff_pad = std::make_shared<v1::Subtract>(full_target, desired);
    const auto pad_begin = std::make_shared<v1::Divide>(diff_pad, two_const);
    const auto pad_end = std::make_shared<v1::Subtract>(diff_pad, pad_begin);

    const auto zero_val = v0::Constant::create(data.get_element_type(), ov::Shape{}, {0});
    const auto out = std::make_shared<v1::Pad>(cropped, pad_begin, pad_end, zero_val, PadMode::CONSTANT);

    return {out};
}
}  // namespace detail

ov::OutputVector center_crop_pad(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);
    const auto inputs = node.get_ov_inputs();
    const auto axes_attr = node.get_attribute_value<std::vector<int64_t>>("axes", {});
    return detail::center_crop_pad_impl(inputs, axes_attr);
}

ONNX_OP("CenterCropPad", OPSET_SINCE(1), ai_onnx::opset_1::center_crop_pad);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
