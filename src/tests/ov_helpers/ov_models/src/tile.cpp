// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeTile(const ov::Output<Node>& in, const std::vector<int64_t>& repeats) {
    auto repeatsNode =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, std::vector<size_t>{repeats.size()}, repeats);
    auto tileNode = std::make_shared<ov::op::v0::Tile>(in, repeatsNode);
    return tileNode;
}

}  // namespace builder
}  // namespace ngraph
