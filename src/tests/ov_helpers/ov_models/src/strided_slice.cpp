// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/strided_slice.hpp"

#include "openvino/op/slice.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makeSlice(const ov::Output<Node>& in,
                                    const std::vector<int64_t>& begin,
                                    const std::vector<int64_t>& end,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& axes,
                                    const element::Type& type) {
    ov::Shape constShape = {begin.size()};
    auto beginNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, begin.data());
    auto endNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, end.data());
    auto strideNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, stride.data());
    if (!axes.empty()) {
        auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, axes.data());
        return std::make_shared<ov::op::v8::Slice>(in, beginNode, endNode, strideNode, axesNode);
    } else {
        return std::make_shared<ov::op::v8::Slice>(in, beginNode, endNode, strideNode);
    }
}

std::shared_ptr<ov::Node> makeSlice(const ov::Output<Node>& in,
                                    const ov::Output<Node>& begin,
                                    const ov::Output<Node>& end,
                                    const ov::Output<Node>& stride,
                                    const ov::Output<Node>& axes) {
    return std::make_shared<ov::op::v8::Slice>(in, begin, end, stride, axes);
}

std::shared_ptr<ov::Node> makeSlice(const ov::Output<Node>& in,
                                    const ov::Output<Node>& begin,
                                    const ov::Output<Node>& end,
                                    const ov::Output<Node>& stride) {
    return std::make_shared<ov::op::v8::Slice>(in, begin, end, stride);
}
}  // namespace builder
}  // namespace ngraph
