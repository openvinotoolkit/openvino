// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeScatterUpdate(const ngraph::Output<Node> &in,
                                                const ngraph::Output<Node> &indices,
                                                const ngraph::Output<Node> &update,
                                                std::size_t axis) {
    auto axis_node = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                std::vector<uint64_t>{axis});
    auto dtsNode = std::make_shared<ngraph::opset3::ScatterUpdate>(in, indices, update, axis_node);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph