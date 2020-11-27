// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeStridedSlice(const ngraph::Output<Node> &in,
                                               const std::vector<int64_t> &begin,
                                               const std::vector<int64_t> &end,
                                               const std::vector<int64_t> &stride,
                                               const element::Type &type,
                                               const std::vector<int64_t> &begin_mask,
                                               const std::vector<int64_t> &end_mask,
                                               const std::vector<int64_t> &new_axis_mask,
                                               const std::vector<int64_t> &shrink_mask,
                                               const std::vector<int64_t> &ellipsis_mask) {
    ngraph::Shape constShape = {begin.size()};
    auto beginNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape, begin.data());
    auto endNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape, end.data());
    auto strideNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape, stride.data());
    auto ssNode = std::make_shared<ngraph::opset2::StridedSlice>(in, beginNode, endNode, strideNode, begin_mask, end_mask,
                                                                 new_axis_mask, shrink_mask, ellipsis_mask);
    return ssNode;
}

}  // namespace builder
}  // namespace ngraph
