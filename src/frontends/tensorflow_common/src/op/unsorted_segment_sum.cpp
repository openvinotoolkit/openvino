// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
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
OutputVector translate_unsorted_segment_sum_op(const NodeContext& node) {
    default_op_checks(node, 3, {"UnsortedSegmentSum"});
    auto data = node.get_input(0);
    auto segment_ids = node.get_input(1);
    auto num_segments = node.get_input(2);

    // convert both segment_ids and num_segments to int64 type
    // since EmbeddingSegmentsSum requires to have them of the same type
    segment_ids = make_shared<v0::Convert>(segment_ids, element::i64);
    num_segments = make_shared<v0::Convert>(num_segments, element::i64);

    // create auxiliary constants
    auto const_zero_i64 = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto const_one_i64 = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto const_one_i64_scalar = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto const_minus_one_i64 = make_shared<v0::Constant>(element::i64, Shape{1}, -1);
    auto data_const_zero = create_same_type_const_scalar<float>(data, 0.0f);

    Output<Node> data_shape = make_shared<v3::ShapeOf>(data, element::i64);

    // adjust data and segment_ids inputs for ND indices
    // to make indices 1D tensor
    // for example, data shape = [2, 3, 4] and segment_ids shape - [2, 3]
    // so they need adjustment to new data shape [6, 4] and segment_ids shape [6]
    auto segment_ids_rank = compute_subgraph_scalar_rank(segment_ids, element::i64, false);
    // 1. segment_ids needs to be flatten
    segment_ids = make_shared<v1::Reshape>(segment_ids, const_minus_one_i64, false);
    // 2. flatten first (segment_ids_rank - 1) dimensions into one dimension
    auto stop_const = make_shared<v0::Constant>(element::i64, Shape{1}, numeric_limits<int64_t>::max());
    auto step_const = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    Output<Node> slice_shape = make_shared<v8::Slice>(data_shape, segment_ids_rank, stop_const, step_const);
    auto new_data_shape = make_shared<v0::Concat>(OutputVector{const_minus_one_i64, slice_shape}, 0);
    data = make_shared<v1::Reshape>(data, new_data_shape, false);

    // segment ids can be negative for which the resulted data will be zeroed
    // so it needs to introduce default slice of zeros in the data
    // 1. create default slice that will be used for negative segment ids
    slice_shape = make_shared<v0::Concat>(OutputVector{const_one_i64, slice_shape}, 0);
    auto default_slice = make_shared<v3::Broadcast>(data_const_zero, slice_shape);
    // 2. update data with the default slice
    data_shape = make_shared<v3::ShapeOf>(data, element::i64);
    data = make_shared<v0::Concat>(OutputVector{data, default_slice}, 0);

    // compute default index
    auto squeeze_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto default_index = get_data_slice(data_shape, 0, 1, 1);
    default_index = make_shared<v0::Squeeze>(default_index, squeeze_axis);

    // adjust segment ids to have zero instead of negative values
    auto is_negative_segment_id = make_shared<v1::Less>(segment_ids, const_zero_i64);
    segment_ids = make_shared<v1::Select>(is_negative_segment_id, const_zero_i64, segment_ids);

    // generate indices input for EmbeddingSegmentSum
    // that will collect slices consequently from data for each segment
    auto segment_ids_shape = make_shared<v3::ShapeOf>(segment_ids, element::i64);
    auto num_indices = make_shared<v0::Squeeze>(segment_ids_shape, squeeze_axis);
    auto indices = make_shared<v4::Range>(const_zero_i64, num_indices, const_one_i64_scalar, element::i64)->output(0);

    // adjust the generated indices to retrieve the default slice for original negative segment ids
    indices = make_shared<v1::Select>(is_negative_segment_id, default_index, indices);

    // since EmbeddingSegmentSum accepts only sorted segments ids
    // it needs to sort them and reorder indices
    auto topk =
        make_shared<v11::TopK>(segment_ids, num_indices, 0, TopKMode::MIN, TopKSortType::SORT_VALUES, element::i32);
    segment_ids = topk->output(0);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    indices = make_shared<v8::Gather>(indices, topk->output(1), gather_axis);

    // compute UnsortedSegmentSum using EmbeddingSegmentSum
    auto unsorted_segment_sum =
        make_shared<v3::EmbeddingSegmentsSum>(data, indices, segment_ids, num_segments, default_index);
    set_node_name(node.get_name(), unsorted_segment_sum);
    return {unsorted_segment_sum};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
