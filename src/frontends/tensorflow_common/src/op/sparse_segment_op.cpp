// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unique.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_sparse_segment_op(const NodeContext& node) {
    default_op_checks(node, 3, {"SparseSegmentMean", "SparseSegmentSqrtN"});
    auto data = node.get_input(0);
    auto op_type = node.get_op_type();
    auto indices = node.get_input(1);
    indices = std::make_shared<v0::Convert>(indices, element::i64);
    auto segment_ids = node.get_input(2);
    segment_ids = std::make_shared<v0::Convert>(segment_ids, element::i64);
    auto data_rank = compute_subgraph_scalar_rank(data, element::i32, false);

    // get the last index from segment_ids
    auto segments_ids_size = std::make_shared<v3::ShapeOf>(segment_ids, element::i64);
    auto const_one = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto const_zero = make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto last_idx = std::make_shared<v1::Subtract>(segments_ids_size, const_one);

    // segment_ids are always sorted
    // so the last index from segment_ids can be used to determine the number of segments
    auto last_segment_idx = std::make_shared<v8::Gather>(segment_ids, last_idx, const_zero);
    auto n_segments = std::make_shared<v1::Add>(last_segment_idx, const_one);

    // get sums of sparse segments
    auto scalar_n_segments = make_shared<v0::Squeeze>(n_segments, const_zero);
    Output<Node> result = make_shared<v3::EmbeddingSegmentsSum>(data, indices, segment_ids, scalar_n_segments);

    // get the sizes of each segment
    auto unique_segment_ids = make_shared<v10::Unique>(segment_ids, true, element::i64, element::i64);
    Output<Node> divisors = make_shared<v3::Broadcast>(const_one, n_segments);
    divisors = make_shared<v3::ScatterUpdate>(divisors,
                                              unique_segment_ids->output(0),
                                              unique_segment_ids->output(3),
                                              const_zero);
    divisors = make_shared<v1::ConvertLike>(divisors, data);

    if (op_type == "SparseSegmentSqrtN") {
        divisors = make_shared<v0::Sqrt>(divisors);
    }

    // since result has a shape [num_segments, s1, ..., st]
    // it must reshape divisors to newshape [num_segments, 1, ..., 1]
    Output<Node> divisors_shape = make_shared<v3::Broadcast>(const_one, data_rank);
    divisors_shape = make_shared<v3::ScatterUpdate>(divisors_shape, const_zero, n_segments, const_zero);
    divisors = std::make_shared<v1::Reshape>(divisors, divisors_shape, false);

    // normalize result by a size of a segment or square-root of it
    result = std::make_shared<v1::Divide>(result, divisors);

    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
