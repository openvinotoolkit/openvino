// Copyright (C) 2018-2024 Intel Corporation
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
    auto complex_type_mark_updates = as_type_ptr<ComplexTypeMark>(updates.get_node_shared_ptr());
    auto zero_scalar = create_same_type_const(updates, 0);

    if (complex_type_mark_updates) {
        updates = complex_type_mark_updates->input_value(0);
        // Add two auxiliary dimensions to the shape tensor
        auto shape_of_op = make_shared<v0::ShapeOf>(updates);
        auto shape_dims = make_shared<v0::ShapeOf>(shape_of_op);
        auto aux_shape = create_same_type_const<int32_t>(shape, std::vector<int32_t>{2}, Shape{1});
        auto updated_shape = make_shared<v0::Concat>(OutputVector{aux_shape, shape_dims}, 0);

        auto input_data = zero_scalar;
        auto broadcast = make_shared<v3::Broadcast>(input_data, updated_shape);

        auto scatter_nd = make_shared<v3::ScatterNDUpdate>(broadcast, input_indices, updates);
        set_node_name(node.get_name(), scatter_nd);

        auto complex_scatter_nd =
            make_shared<ComplexTypeMark>(scatter_nd, complex_type_mark_updates->get_complex_part_type());
        return {complex_scatter_nd};
    }

    auto input_data = zero_scalar;
    auto broadcast = make_shared<v3::Broadcast>(input_data, shape);
    auto scatter_nd = make_shared<v3::ScatterNDUpdate>(broadcast, input_indices, updates);
    set_node_name(node.get_name(), scatter_nd);
    return {scatter_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
