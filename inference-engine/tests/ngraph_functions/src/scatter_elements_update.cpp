// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeScatterElementsUpdate(const ngraph::Output<Node> &in,
                                                        const ngraph::Output<Node> &indices,
                                                        const ngraph::Output<Node> &update,
                                                        std::size_t axis) {
    auto axis_node = default_opset::Constant::create(element::i64, Shape{}, {axis});
    auto dtsNode = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(in, indices, update, axis_node);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph