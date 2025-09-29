// Copyright (C) 2018-2025 Intel Corporation
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
ov::OutputVector argmin(const ov::frontend::onnx::Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_min()};
}

ONNX_OP("ArgMin", {1, 11}, ai_onnx::opset_1::argmin);
}  // namespace opset_1

namespace opset_12 {
ov::OutputVector argmin(const ov::frontend::onnx::Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_min()};
}

ONNX_OP("ArgMin", OPSET_SINCE(12), ai_onnx::opset_12::argmin);
}  // namespace opset_12
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
