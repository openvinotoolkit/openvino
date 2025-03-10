// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_scatter_nd_op(const NodeContext& node) {
    default_op_checks(node, 3, {"ScatterNd", "SCATTER_ND"}, true);
    auto input_indices = node.get_input(0);
    auto updates = node.get_input(1);
    auto shape = node.get_input(2);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(updates.get_node_shared_ptr());

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        updates = complex_type_mark->get_data();

        auto new_dim = create_same_type_const<int32_t>(shape, vector<int32_t>{2}, Shape{1});
        auto new_shape = make_shared<v0::Concat>(OutputVector{shape, new_dim}, -1);

        auto const_zero = create_same_type_const<int32_t>(updates, vector<int32_t>{0}, Shape{1});
        auto broadcast = make_shared<v3::Broadcast>(const_zero, new_shape);

        auto complex_scatter_nd = make_shared<v3::ScatterNDUpdate>(broadcast, input_indices, updates);

        set_node_name(node.get_name(), complex_scatter_nd);
        auto complex_result = make_shared<ComplexTypeMark>(complex_scatter_nd, complex_part_type);

        return {complex_result};
    }

    auto input_data = create_same_type_const<int32_t>(updates, vector<int32_t>{0}, Shape{1});
    auto broadcast = make_shared<v3::Broadcast>(input_data, shape);
    auto scatter_nd = make_shared<v3::ScatterNDUpdate>(broadcast, input_indices, updates);
    set_node_name(node.get_name(), scatter_nd);
    return {scatter_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
