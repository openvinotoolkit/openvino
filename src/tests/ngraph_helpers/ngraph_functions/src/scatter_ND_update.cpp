// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeScatterNDUpdate(const ov::Output<ov::Node>& in,
                                              const ov::element::Type& indicesType,
                                              const std::vector<size_t>& indicesShape,
                                              const std::vector<size_t>& indices,
                                              const ov::Output<Node>& update) {
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesType, indicesShape, indices);
    auto dtsNode = std::make_shared<ov::op::v3::ScatterNDUpdate>(in, indicesNode, update);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph
