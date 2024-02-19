// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/average_pool.hpp"

#include "utils/pooling_factory.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector average_pool(const ov::frontend::onnx::Node& node) {
    return pooling::PoolingFactory(node).make_avg_pool();
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
