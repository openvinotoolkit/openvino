// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "utils/pooling_factory.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
    namespace opset_1 {
        ov::OutputVector average_pool(const Node& node) {
            return pooling::PoolingFactory(node).make_avg_pool();
        }
    }
    namespace opset_7 {
        ov::OutputVector average_pool(const Node& node) {
            return pooling::PoolingFactory(node).make_avg_pool_opset7(); 
        }
    }

// Registering the different versions
ONNX_OP("AveragePool", OPSET_SINCE(1), ai_onnx::opset_1::average_pool);
ONNX_OP("AveragePool", OPSET_SINCE(7), ai_onnx::opset_7::average_pool);
// We map 10, 11, and 19 to the newer logic as well
ONNX_OP("AveragePool", OPSET_SINCE(10), ai_onnx::opset_7::average_pool);
ONNX_OP("AveragePool", OPSET_SINCE(11), ai_onnx::opset_7::average_pool);
ONNX_OP("AveragePool", OPSET_SINCE(19), ai_onnx::opset_7::average_pool);

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov