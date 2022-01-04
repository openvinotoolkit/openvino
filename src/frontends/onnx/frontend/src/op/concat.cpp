// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/concat.hpp"

#include <cstdint>

#include "default_opset.hpp"
#include "exceptions.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector concat(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    std::int64_t axis = node.get_attribute_value<std::int64_t>("axis");
    return {std::make_shared<default_opset::Concat>(inputs, axis)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
