// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/normalize_l2.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makeNormalizeL2(const ov::Output<Node>& data,
                                          const std::vector<int64_t>& axes,
                                          float eps,
                                          ov::op::EpsMode epsMode) {
    auto normAxes = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{axes.size()}, axes);
    return std::make_shared<ov::op::v0::NormalizeL2>(data, normAxes, eps, epsMode);
}
}  // namespace builder
}  // namespace ngraph
