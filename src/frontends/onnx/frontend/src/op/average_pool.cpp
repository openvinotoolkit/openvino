// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "utils/pooling_factory.hpp"
namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector average_pool(const ov::frontend::onnx::Node& node) {
    return pooling::PoolingFactory(node).make_avg_pool();
}

ONNX_OP("AveragePool", OPSET_SINCE(1), ai_onnx::opset_1::average_pool);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
