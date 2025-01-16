// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/topk.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
NamedOutputVector translate_top_k_base_op(const NodeContext& node,
                                          const ov::Output<ov::Node>& k_input,
                                          int min_input_size,
                                          const ov::element::Type& index_type = ov::element::i32) {
    default_op_checks(node, min_input_size, {"TopK", "TopKV2", "TOPK_V2"});
    auto input = node.get_input(0);

    // retrieve k attribute
    bool sorted = node.get_attribute<bool>("sorted", true);
    auto topk_index_type = index_type;
    if (index_type == ov::element::i16) {
        // v11::TopK supports only int32 and int64 output index type
        topk_index_type = ov::element::i32;
    }
    auto top_k = make_shared<v11::TopK>(input,
                                        k_input,
                                        -1,
                                        ov::op::v11::TopK::Mode::MAX,
                                        sorted ? v11::TopK::SortType::SORT_VALUES : v11::TopK::SortType::SORT_INDICES,
                                        topk_index_type,
                                        true);
    auto values = top_k->output(0);
    auto indices = top_k->output(1);
    if (index_type != topk_index_type) {
        // satisfy the requested output index type
        indices = make_shared<v0::Convert>(indices, index_type)->output(0);
    }
    set_node_name(node.get_name(), top_k);
    return {{"values", values}, {"indices", indices}};
}

NamedOutputVector translate_top_k_op(const NodeContext& node) {
    // retrieve k attribute
    auto k = node.get_attribute<int64_t>("k");
    auto k_input = make_shared<v0::Constant>(ov::element::i64, Shape{}, std::vector<int64_t>({k}));
    return translate_top_k_base_op(node, k_input, 1);
}

NamedOutputVector translate_top_k_v2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TopKV2", "TOPK_V2"});
    auto index_type = node.get_attribute<ov::element::Type>("index_type", ov::element::i32);
    auto k_input = node.get_input(1);
    return translate_top_k_base_op(node, k_input, 1, index_type);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
