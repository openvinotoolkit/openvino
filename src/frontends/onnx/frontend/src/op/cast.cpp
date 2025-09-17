// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector cast(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    int64_t target_type = node.get_attribute_value<int64_t>("to");
    ov::element::Type elem_type = common::get_ov_element_type(target_type);

    // onnx's cast op is using no_clamp and rounding cast.
    // check
    // https://github.com/microsoft/onnxruntime/blob/bac0bff72b1b4e6fd68ae759a32644defac61944/onnxruntime/test/providers/cpu/tensor/cast_op_test.cc#L959
    // for example, float to int4, input value 31.9
    //   onnx cast:                               31.9 -> 32 -> 0x20 -> 0 (round and no_clamp)
    //   ov convert - default:                    31.9 -> 31 -> 7         (trunc and clamp)
    // so here we use ov::op::v0::Convert with no_clamp=true and use_rounding=true
    // to align with onnx cast op behavior.
    return {std::make_shared<v0::Convert>(data, elem_type, true, true)};
}

ONNX_OP("Cast", OPSET_SINCE(1), ai_onnx::opset_1::cast);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
