// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include "openvino/op/constant.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeScatterElementsUpdate(const ov::Output<Node>& in,
                                                    const element::Type& indicesType,
                                                    const std::vector<size_t>& indicesShape,
                                                    const std::vector<size_t>& indices,
                                                    const ov::Output<Node>& update,
                                                    int axis) {
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesType, indicesShape, indices);
    auto axis_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{}, std::vector<int>{axis});
    auto dtsNode = std::make_shared<ov::op::v3::ScatterElementsUpdate>(in, indicesNode, update, axis_node);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph
