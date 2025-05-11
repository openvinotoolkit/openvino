// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace opset8;
using namespace element;

NamedOutputs embedding(const NodeContext& node) {
    auto data_ids = node.get_input("Ids");
    auto data_w = node.get_input("W");

    auto padding_idx = node.get_attribute<int64_t>("padding_idx");

    const auto const_axis0 = Constant::create<int32_t>(i64, {1}, {0});

    std::shared_ptr<Node> node_embedding;
    if (padding_idx < 0)  // no mask
    {
        node_embedding = std::make_shared<Gather>(data_w, data_ids, const_axis0);
    } else {  // mask embedding
        auto node_shape_of_w = std::make_shared<ShapeOf>(data_w);
        auto node_vocab_size = std::make_shared<Gather>(node_shape_of_w,
                                                        Constant::create<int64_t>(i64, {1}, {0}),
                                                        const_axis0);  // vocab_size
        auto node_stop = std::make_shared<Squeeze>(node_vocab_size);

        auto node_range = std::make_shared<Range>(Constant::create<int64_t>(i64, {}, {0}),
                                                  node_stop,
                                                  Constant::create<int64_t>(i64, {}, {1}),
                                                  i64);

        auto node_equal = std::make_shared<Equal>(node_range, Constant::create(i64, {1}, {padding_idx}));
        auto node_mask = std::make_shared<Unsqueeze>(node_equal, Constant::create<int64_t>(i64, {1}, {1}));

        data_w = std::make_shared<Select>(node_mask,
                                          Constant::create<float>(f32, {1}, {0}),
                                          data_w,
                                          ov::op::AutoBroadcastType::NUMPY);  // masked W

        node_embedding = std::make_shared<Gather>(data_w, data_ids, const_axis0);
    }

    return node.default_single_output_mapping({node_embedding}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
