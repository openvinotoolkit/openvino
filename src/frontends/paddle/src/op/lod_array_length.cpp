// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs lod_array_length(const NodeContext& node) {
    using namespace default_opset;
    const auto x = node.get_input("X");
    const auto shape = std::make_shared<default_opset::ShapeOf>(x);
    // here simply get the length along the concated axis.
    // we've lost the original tensor array length actually.
    // luckily it's equalent since all elements are concated.
    const auto const_1_node = Constant::create(element::i64, {1}, {1});
    const auto const_2_node = Constant::create(element::i64, {1}, {2});
    const auto len = std::make_shared<StridedSlice>(shape,
                                                    const_1_node,
                                                    const_2_node,
                                                    std::vector<int64_t>{0},
                                                    std::vector<int64_t>{0});

    return node.default_single_output_mapping({len}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
