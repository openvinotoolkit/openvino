// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"
namespace {
ov::op::PadMode get_pad_mode(std::string mode) {
    ov::op::PadMode pad_mode;

    if (mode == "constant") {
        pad_mode = ov::op::PadMode::CONSTANT;
    } else if (mode == "reflect") {
        pad_mode = ov::op::PadMode::REFLECT;
    } else if (mode == "edge") {
        pad_mode = ov::op::PadMode::EDGE;
    } else {
        OPENVINO_THROW("Unsupported padding mode: [" + mode + "]");
    }

    return pad_mode;
}
}  // namespace
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector pad(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);

    const auto data_rank = node.get_ov_inputs().at(0).get_partial_shape().rank();
    CHECK_VALID_NODE(node, data_rank.is_static(), "Data rank must be static for pad op");
    const auto data_rank_value = data_rank.get_length();

    double value = node.get_attribute_value<double>("value", 0);
    const std::string mode = node.get_attribute_value<std::string>("mode", "constant");
    ov::op::PadMode pad_mode = get_pad_mode(mode);

    const auto paddings = convpool::get_pads(node, data_rank_value);
    ov::CoordinateDiff padding_below = paddings.first;
    ov::CoordinateDiff padding_above = paddings.second;

    return {std::make_shared<v12::Pad>(
        data,
        std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{padding_below.size()}, padding_below),
        std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{padding_above.size()}, padding_above),
        std::make_shared<v0::Constant>(data.get_element_type(), ov::Shape{}, std::vector<double>{value}),
        pad_mode)};
}

ONNX_OP("Pad", OPSET_RANGE(1, 10), ai_onnx::opset_1::pad);
}  // namespace opset_1
namespace opset_11 {
ov::OutputVector pad(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs[0];
    const auto& pads = inputs[1];
    ov::Output<ov::Node> values;
    ov::Output<ov::Node> padding_begin;
    ov::Output<ov::Node> padding_end;

    if (inputs.size() == 3 && !ov::op::util::is_null(inputs[2])) {
        values = reshape::interpret_as_scalar(inputs[2]);
    } else {
        values = v0::Constant::create(data.get_element_type(), ov::Shape{}, {0});
    }

    if (ov::op::util::is_constant(pads.get_node())) {
        std::vector<std::int64_t> pads_vector =
            ov::as_type_ptr<v0::Constant>(pads.get_node_shared_ptr())->get_vector<std::int64_t>();

        std::size_t const half_size = pads_vector.size() / 2;
        std::vector<std::int64_t> padding_begin_values(pads_vector.begin(), pads_vector.begin() + half_size);
        std::vector<std::int64_t> padding_end_values(pads_vector.begin() + half_size, pads_vector.end());

        padding_begin = v0::Constant::create(ov::element::i64, ov::Shape{half_size}, padding_begin_values);
        padding_end = v0::Constant::create(ov::element::i64, ov::Shape{half_size}, padding_end_values);
    } else {
        ov::OutputVector padding = ov::op::util::make_split(pads, 2, 0);

        padding_begin = padding.at(0);
        padding_end = padding.at(1);
    }

    const std::string mode = node.get_attribute_value<std::string>("mode", "constant");
    ov::op::PadMode pad_mode = get_pad_mode(mode);

    return {std::make_shared<v12::Pad>(data, padding_begin, padding_end, values, pad_mode)};
}

ONNX_OP("Pad", OPSET_SINCE(11), ai_onnx::opset_11::pad);
}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
