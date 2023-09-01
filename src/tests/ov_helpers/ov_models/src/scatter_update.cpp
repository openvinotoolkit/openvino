// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeScatterUpdate(const ov::Output<Node> &in,
                                                const element::Type& indicesType,
                                                const std::vector<size_t> &indicesShape,
                                                const std::vector<int64_t> &indices,
                                                const ov::Output<Node> &update,
                                                int64_t axis) {
    auto indicesNode = std::make_shared<ov::opset1::Constant>(indicesType, indicesShape, indices);
    auto axis_node = std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64, ov::Shape{},
                                                                std::vector<int64_t>{axis});
    auto dtsNode = std::make_shared<ov::opset3::ScatterUpdate>(in, indicesNode, update, axis_node);
    return dtsNode;
}

}  // namespace builder
}  // namespace ov
