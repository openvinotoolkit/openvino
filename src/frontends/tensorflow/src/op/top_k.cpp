// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_top_k_v2_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto k = node.get_input(1);

    TENSORFLOW_OP_VALIDATION(node, input.get_partial_shape().rank().is_static(), "Input rank must be static.");
    TENSORFLOW_OP_VALIDATION(node,
                             input.get_partial_shape().rank().get_length() >= 1,
                             "Input rank must be greater than 0.");
    // axis along which to compute top k indices
    int64_t k_axis = input.get_partial_shape().rank().get_length() - 1;
    bool sorted = node.get_attribute<bool>("sorted", true);
    auto res = std::make_shared<TopK>(input,
                                      k,
                                      k_axis,
                                      TopK::Mode::MAX,
                                      sorted ? TopK::SortType::SORT_VALUES : TopK::SortType::SORT_INDICES);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov