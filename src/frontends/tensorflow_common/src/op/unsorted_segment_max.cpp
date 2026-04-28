// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/segment_max.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_unsorted_segment_max_op(const NodeContext& node) {
    default_op_checks(node, 3, {"UnsortedSegmentMax"});
    auto data = node.get_input(0);
    auto segment_ids = node.get_input(1);
    auto num_segments = node.get_input(2);

    // Flatten segment_ids to 1D for TopK
    auto const_minus_one = make_shared<v0::Constant>(element::i64, Shape{1}, vector<int64_t>{-1});
    auto flat_ids = make_shared<v1::Reshape>(segment_ids, const_minus_one, false);

    // Determine K = number of elements
    auto squeeze_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{0});
    auto ids_shape = make_shared<v3::ShapeOf>(flat_ids, element::i64);
    auto num_indices = make_shared<v0::Squeeze>(ids_shape, squeeze_axis);

    // Sort segment_ids ascending via TopK(MIN, SORT_VALUES)
    auto topk = make_shared<v11::TopK>(flat_ids,
                                       num_indices,
                                       /*axis=*/0,
                                       v11::TopK::Mode::MIN,
                                       v11::TopK::SortType::SORT_VALUES,
                                       element::i32);
    auto sorted_segment_ids = topk->output(0);
    auto sort_permutation = topk->output(1);

    // Gather data in sorted order along axis 0
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{0});
    auto sorted_data = make_shared<v8::Gather>(data, sort_permutation, gather_axis);

    // Make num_segments a scalar i64 for SegmentMax
    auto scalar_shape = make_shared<v0::Constant>(element::i32, Shape{0}, vector<int32_t>{});
    auto num_segments_scalar =
        make_shared<v1::Reshape>(make_shared<v0::Convert>(num_segments, element::i64), scalar_shape, false);

    // Cast sorted segment ids to i32
    auto sorted_ids_i32 = make_shared<v0::Convert>(sorted_segment_ids, element::i32);

    // SegmentMax with FillMode::LOWEST: empty segments get the type's minimum value, matching TF
    auto seg_max =
        make_shared<v16::SegmentMax>(sorted_data, sorted_ids_i32, num_segments_scalar, FillMode::LOWEST);
    set_node_name(node.get_name(), seg_max);
    return {seg_max};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
