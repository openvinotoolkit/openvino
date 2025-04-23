// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_gather(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    Output<Node> inputs = context.get_input(0);
    Output<Node> indices = context.get_input(1);

    indices = context.mark_node(std::make_shared<v0::Convert>(indices, element::i64));

    int64_t axis_val;

    try {
        axis_val = context.const_named_param<int64_t>("axis");
    } catch (const ov::Exception& e) {
        // Fallback: get axis from start_index_map
        auto gather_attrs = context.get_attrs();
        auto start_index_map = gather_attrs["start_index_map"].as<std::vector<int64_t>>();
        FRONT_END_GENERAL_CHECK(start_index_map.size() == 1,
                                "Only single-axis gather is supported in fallback.");
        axis_val = start_index_map[0];
    }

    auto axis_node = std::make_shared<v0::Constant>(element::i64, Shape{}, axis_val);
    Output<Node> res = context.mark_node(std::make_shared<v1::Gather>(inputs, indices, axis_node));
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov