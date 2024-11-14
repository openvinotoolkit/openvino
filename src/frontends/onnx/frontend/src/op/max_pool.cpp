// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/util/log.hpp"
#include "utils/pooling_factory.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector max_pool(const ov::frontend::onnx::Node& node) {
    if (node.get_outputs_size() > 1) {
        OPENVINO_WARN("MaxPool: Indices output is not supported and was ignored");
    }
    auto max_pool = pooling::PoolingFactory(node).make_max_pool();
    max_pool.emplace_back(std::make_shared<NullNode>());  // Indices (optional)
    return max_pool;
}

ONNX_OP("MaxPool", OPSET_RANGE(1, 7), ai_onnx::opset_1::max_pool);
}  // namespace opset_1

namespace opset_8 {
ov::OutputVector max_pool(const ov::frontend::onnx::Node& node) {
    return pooling::PoolingFactory(node).make_max_pool_with_indices();
}
ONNX_OP("MaxPool", OPSET_SINCE(8), ai_onnx::opset_8::max_pool);
}  // namespace opset_8
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
