// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <algorithm>

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_slice_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Slice", "SLICE"});
    auto input = node.get_input(0);
    auto start = node.get_input(1);
    auto size = node.get_input(2);

    // create axiliary constants
    auto const_one = create_same_type_const_scalar<int32_t>(start, 1);

    // compute stop values in case non-negative sizes
    auto stop_pos = make_shared<v1::Add>(start, size);

    // When `size` is a Constant with no negative entries, the negative-size
    // branch is dead and can be elided. Emitting `stop = start + size` directly
    // (without the ShapeOf + ConvertLike + Less + Select cascade that handles
    // size = -1) lets Slice's shape inference produce a static output whenever
    // `start` is also constant — which is the common TFLite case.
    auto size_const = ov::as_type_ptr<v0::Constant>(size.get_node_shared_ptr());
    bool size_is_nonneg_constant = false;
    if (size_const) {
        const auto size_values = size_const->cast_vector<int64_t>();
        size_is_nonneg_constant =
            std::all_of(size_values.begin(), size_values.end(), [](int64_t v) { return v >= 0; });
    }

    Output<Node> stop;
    if (size_is_nonneg_constant) {
        stop = stop_pos;
    } else {
        // compute stop values in case negative sizes
        // since TensorFlow supports only -1 among negative sizes
        // assign stop values to the data shape
        auto const_zero = create_same_type_const_scalar<int32_t>(start, 0);
        Output<Node> stop_neg = make_shared<v3::ShapeOf>(input);
        stop_neg = make_shared<v1::ConvertLike>(stop_neg, size);

        // select the correct stop value based on a sign of size value.
        // FloorMod(size, input_shape) would *not* be an equivalent simplification:
        // for size = -1 it yields input_shape - 1, off by one from the required
        // input_shape; the smallest correct FloorMod rewrite
        // (start + FloorMod(size, input_shape - start + 1)) needs more ops than
        // this Select and has no bounds tracking, so it offers no benefit.
        auto negative_sizes_mask = make_shared<v1::Less>(size, const_zero);
        stop = make_shared<v1::Select>(negative_sizes_mask, stop_neg, stop_pos);
    }

    // broadcast step value
    auto start_shape = make_shared<v3::ShapeOf>(start);
    auto step = make_shared<v3::Broadcast>(const_one, start_shape);

    auto res = make_shared<v8::Slice>(input, start, stop, step);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
