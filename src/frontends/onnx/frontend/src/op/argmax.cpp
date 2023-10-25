// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/argmax.hpp"

#include "exceptions.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/arg_min_max_factory.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector argmax(const Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_max()};
}

}  // namespace set_1

namespace set_12 {
OutputVector argmax(const Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_max()};
}

}  // namespace set_12

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
