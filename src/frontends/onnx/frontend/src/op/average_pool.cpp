// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/average_pool.hpp"

#include "ngraph/node.hpp"
#include "utils/pooling_factory.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector average_pool(const Node& node) {
    return pooling::PoolingFactory(node).make_avg_pool();
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
