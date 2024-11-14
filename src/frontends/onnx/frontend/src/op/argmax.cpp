// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/arg_min_max_factory.hpp"
namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector argmax(const ov::frontend::onnx::Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_max()};
}

ONNX_OP("ArgMax", OPSET_RANGE(1, 11), ai_onnx::opset_1::argmax);
}  // namespace opset_1

namespace opset_12 {
ov::OutputVector argmax(const ov::frontend::onnx::Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_max()};
}

ONNX_OP("ArgMax", OPSET_SINCE(12), ai_onnx::opset_12::argmax);
}  // namespace opset_12
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
