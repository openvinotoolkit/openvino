// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateTopKV2Op(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto k = node.get_ng_input(1);

    TF_OP_VALIDATION_CHECK(node, input.get_partial_shape().rank().is_static(), "Input rank must be static.");
    TF_OP_VALIDATION_CHECK(node, input.get_partial_shape().rank().get_length() >= 1, "Input rank must be greater than 0.");
    // axis along which to compute top k indices
    int64_t k_axis = input.get_partial_shape().rank().get_length() - 1;
    bool sorted = node.get_attribute<bool>("sorted", true);
    auto top_k = std::make_shared<TopK>(input,
                                        k,
                                        k_axis,
                                        TopK::Mode::MAX,
                                        sorted ? TopK::SortType::SORT_VALUES : TopK::SortType::SORT_INDICES);
    top_k->set_friendly_name(node.get_name());
    return top_k->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph