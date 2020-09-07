// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeTile(const ngraph::Output<Node>& in,
                                       const std::vector<size_t>& repeats) {
    auto repeatsNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, std::vector<size_t>{repeats.size()}, repeats);
    auto tileNode = std::make_shared<ngraph::opset1::Tile>(in, repeatsNode);
    return tileNode;
}

}  // namespace builder
}  // namespace ngraph
