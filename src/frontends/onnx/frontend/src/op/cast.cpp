// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/cast.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/type/element_type.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector cast(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    int64_t target_type = node.get_attribute_value<int64_t>("to");
    element::Type elem_type = common::get_ngraph_element_type(target_type);

    return {std::make_shared<default_opset::Convert>(data, elem_type)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
