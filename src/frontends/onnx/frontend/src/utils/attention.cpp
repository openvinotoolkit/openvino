// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/attention.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace attention {

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<v3::ShapeOf>(node), dims);
}

}  // namespace attention
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
