// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_arg_min_max(const NodeContext& node, std::string mode) {
    default_op_checks(node, 1, {"ArgMax", "ArgMin", "ARG_MAX", "ARG_MIN"});
    auto input = node.get_input(0);

    // TensorFlow uses axis with default value equal to zero
    int64_t axis = 0;
    if (node.get_input_size() > 1) {
        TENSORFLOW_OP_VALIDATION(node,
                                 as_type_ptr<v0::Constant>(node.get_input(1).get_node_shared_ptr()),
                                 "ArgMax/ArgMin is not supported with non-constant axis input");
        std::vector<int64_t> axes;
        get_const_input(node, 1, &axes);
        TENSORFLOW_OP_VALIDATION(node, axes.size() == 1, "ArgMax/ArgMin must be with a scalar axis input.");
        axis = axes[0];
    }
    auto output_type = node.get_attribute<element::Type>("output_type", element::i64);
    auto topk_output_type = output_type;
    if (topk_output_type != element::i32 && topk_output_type != element::i64) {
        // OV TopK supports only element::i32 and element::i64
        // so use temporarily element::i64
        topk_output_type = element::i64;
    }

    // compute indices of max/min values using TopK
    auto k = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto top_k_mode = (mode == "max" ? v11::TopK::Mode::MAX : v11::TopK::Mode::MIN);
    auto sort_type = v11::TopK::SortType::SORT_VALUES;
    auto top_k = make_shared<v11::TopK>(input, k, axis, top_k_mode, sort_type, topk_output_type, true);

    auto axis_to_remove = make_shared<v0::Constant>(element::i64, Shape{1}, vector<int64_t>({axis}));
    auto res = make_shared<v0::Squeeze>(top_k->output(1), axis_to_remove)->output(0);
    if (topk_output_type != output_type) {
        // use required type for the output
        res = make_shared<v0::Convert>(res, output_type);
    }
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}

OutputVector translate_arg_max_op(const NodeContext& node) {
    return (translate_arg_min_max(node, "max"));
}

OutputVector translate_arg_min_op(const NodeContext& node) {
    return (translate_arg_min_max(node, "min"));
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
