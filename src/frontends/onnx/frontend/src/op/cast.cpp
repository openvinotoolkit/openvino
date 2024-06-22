// Copyright (C) 2018-2024 Intel Corporation
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

    return {std::make_shared<v0::Convert>(data, elem_type)};
}

static bool registered = register_translator("Cast", VersionRange::single_version_for_all_opsets(), cast);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
