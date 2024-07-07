// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/pad.hpp"

#include "core/null_node.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/op_types.hpp"
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
namespace op {
namespace custom {
namespace set_1 {
ov::OutputVector pad(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs[0];
    const auto& pads_input = inputs[1];
    auto pads = pads_input;
    const auto& pads_pshape = pads.get_partial_shape();
    if (pads_pshape.rank().is_static() && pads_pshape.size() == 2) {
        pads = std::make_shared<v0::Squeeze>(pads);
    }
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
}  // namespace set_1
}  // namespace custom
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
