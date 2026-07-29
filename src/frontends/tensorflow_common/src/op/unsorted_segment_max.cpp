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

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_unsorted_segment_max_op(const NodeContext& node) {
    default_op_checks(node, 3, {"UnsortedSegmentMax"});
    auto data = node.get_input(0);
    auto segment_ids = node.get_input(1);
    auto num_segments = node.get_input(2);

    auto const_minus_one =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto flat_ids = std::make_shared<ov::op::v1::Reshape>(segment_ids, const_minus_one, false);

    auto squeeze_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(flat_ids, ov::element::i64);
    auto num_indices = std::make_shared<ov::op::v0::Squeeze>(ids_shape, squeeze_axis);

    auto topk = std::make_shared<ov::op::v11::TopK>(flat_ids,
                                                    num_indices,
                                                    0,
                                                    ov::op::v11::TopK::Mode::MIN,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i32);
    auto sorted_segment_ids = topk->output(0);
    auto sort_permutation = topk->output(1);

    auto gather_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto sorted_data = std::make_shared<ov::op::v8::Gather>(data, sort_permutation, gather_axis);

    auto scalar_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
    auto num_segments_scalar =
        std::make_shared<ov::op::v1::Reshape>(std::make_shared<ov::op::v0::Convert>(num_segments, ov::element::i64),
                                              scalar_shape,
                                              false);

    auto sorted_ids_i32 = std::make_shared<ov::op::v0::Convert>(sorted_segment_ids, ov::element::i32);

    auto seg_max = std::make_shared<ov::op::v16::SegmentMax>(sorted_data,
                                                             sorted_ids_i32,
                                                             num_segments_scalar,
                                                             ov::op::FillMode::LOWEST);
    set_node_name(node.get_name(), seg_max);
    return {seg_max};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
