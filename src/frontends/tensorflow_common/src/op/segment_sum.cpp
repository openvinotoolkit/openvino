// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset9;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_segment_sum_op(const NodeContext& node) {
    default_op_checks(node, 2, {"SegmentSum"});
    auto data = node.get_input(0);
    auto segment_ids = node.get_input(1);

    // compute SegmentSum using EmbeddingSegmentSum
    // for this prepare all the required inputs
    auto indices_type = segment_ids.get_element_type();
    // 1. compute a number of segments using segment_ids values
    // do not forget that segment ids are counting from zero
    auto reduction_axis = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto num_segments_minus1 = make_shared<ReduceMax>(segment_ids, reduction_axis, false);
    auto one = make_shared<Constant>(indices_type, Shape{}, 1);
    auto num_segments = make_shared<Add>(num_segments_minus1, one);

    // 2. generate indices input for EmbeddingSegmentSum
    // that will collect slices consequently from data for each segment
    auto squeeze_axis = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto segment_ids_shape = make_shared<ShapeOf>(segment_ids, indices_type);
    auto num_indices = make_shared<Squeeze>(segment_ids_shape, squeeze_axis);
    auto indices = make_shared<Range>(make_shared<Constant>(indices_type, ov::Shape{}, 0),
                                      num_indices,
                                      make_shared<Constant>(indices_type, ov::Shape{}, 1),
                                      indices_type);

    auto emb_segment_sum = make_shared<EmbeddingSegmentsSum>(data, indices, segment_ids, num_segments);
    set_node_name(node.get_name(), emb_segment_sum);
    return {emb_segment_sum};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov