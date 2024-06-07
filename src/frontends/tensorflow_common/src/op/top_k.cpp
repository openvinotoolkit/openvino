// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/topk.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
NamedOutputVector translate_top_k_base_op(const NodeContext& node,
                                          const ov::Output<ov::Node>& k_input,
                                          int min_input_size) {
    default_op_checks(node, min_input_size, {"TopK", "TopKV2", "TOPK_V2"});
    auto input = node.get_input(0);

    // retrieve k attribute
    bool sorted = node.get_attribute<bool>("sorted", true);
    auto top_k = make_shared<v11::TopK>(input,
                                        k_input,
                                        -1,
                                        ov::op::v11::TopK::Mode::MAX,
                                        sorted ? v11::TopK::SortType::SORT_VALUES : v11::TopK::SortType::SORT_INDICES,
                                        ov::element::i32,
                                        true);
    set_node_name(node.get_name(), top_k);
    return {{"values", top_k->output(0)}, {"indices", top_k->output(1)}};
}
NamedOutputVector translate_top_k_op(const NodeContext& node) {
    // retrieve k attribute
    auto k = node.get_attribute<int64_t>("k");
    auto k_input = make_shared<v0::Constant>(ov::element::i64, Shape{}, std::vector<int64_t>({k}));
    return translate_top_k_base_op(node, k_input, 1);
}

NamedOutputVector translate_top_k_v2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TopKV2", "TOPK_V2"});
    auto k_input = node.get_input(1);
    return translate_top_k_base_op(node, k_input, 1);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
