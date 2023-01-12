// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_dim(NodeContext& context) {
    auto shape = std::make_shared<opset8::ShapeOf>(context.get_input(0), element::i32);
    auto rank = std::make_shared<opset8::ShapeOf>(shape, element::i32);
    auto squeeze = std::make_shared<opset8::Squeeze>(rank);
    context.mark_nodes({shape, rank, squeeze});
    return squeeze->outputs();
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov