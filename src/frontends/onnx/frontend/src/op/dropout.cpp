// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/op_types.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {
ov::OutputVector build_dropout(const ov::frontend::onnx::Node& node, bool training_mode) {
    CHECK_VALID_NODE(node, !training_mode, "Training mode is not supported for Dropout op");

    const auto input_data = node.get_ov_inputs().at(0);
    const bool return_mask = node.get_outputs_size() > 1;

    if (return_mask) {
        const auto mask =
            std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::boolean, ov::Shape{}, {true}),
                                            std::make_shared<v3::ShapeOf>(input_data));
        return {input_data, mask};
    } else {
        return {input_data};
    }
}
}  // namespace

namespace opset_12 {
ov::OutputVector dropout(const ov::frontend::onnx::Node& node) {
    const auto ng_inputs = node.get_ov_inputs();
    // seed attribute and ratio input are ignored because traning mode is not
    // supported anyway
    bool training_mode = false;  // default value
    if (ng_inputs.size() > 2 && !ov::op::util::is_null(ng_inputs.at(2))) {
        CHECK_VALID_NODE(node,
                         ov::op::util::is_constant(ng_inputs.at(2).get_node_shared_ptr()),
                         "Non-constant training_mode input is not supported.");
        training_mode = ov::as_type_ptr<v0::Constant>(ng_inputs.at(2).get_node_shared_ptr())->cast_vector<bool>()[0];
    }
    return build_dropout(node, training_mode);
}
ONNX_OP("Dropout", OPSET_SINCE(12), ai_onnx::opset_12::dropout);
}  // namespace opset_12

namespace opset_7 {
ov::OutputVector dropout(const ov::frontend::onnx::Node& node) {
    // "is_test" attribute was removed
    // ratio attribute is ignored because traning mode is not supported
    const bool training_mode = false;

    return build_dropout(node, training_mode);
}
ONNX_OP("Dropout", OPSET_RANGE(7, 11), ai_onnx::opset_7::dropout);
}  // namespace opset_7

namespace opset_1 {
ov::OutputVector dropout(const ov::frontend::onnx::Node& node) {
    // legacy consumed_inputs attribute ignored
    // ratio attribute is ignored because traning mode is not supported
    const bool training_mode = !node.get_attribute_value<int64_t>("is_test", 0);

    return build_dropout(node, training_mode);
}
ONNX_OP("Dropout", OPSET_RANGE(1, 6), ai_onnx::opset_1::dropout);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
