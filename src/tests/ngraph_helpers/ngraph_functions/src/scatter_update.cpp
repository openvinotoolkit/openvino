// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeScatterUpdate(const ngraph::Output<Node> &in,
                                                const element::Type& indicesType,
                                                const std::vector<size_t> &indicesShape,
                                                const std::vector<int64_t> &indices,
                                                const ngraph::Output<Node> &update,
                                                int64_t axis) {
    auto indicesNode = std::make_shared<ngraph::opset1::Constant>(indicesType, indicesShape, indices);
    auto axis_node = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                std::vector<int64_t>{axis});
    auto dtsNode = std::make_shared<ngraph::opset3::ScatterUpdate>(in, indicesNode, update, axis_node);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph