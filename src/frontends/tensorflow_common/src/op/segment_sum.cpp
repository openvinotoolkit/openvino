// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_segment_sum_op(const NodeContext& node) {
    default_op_checks(node, 2, {"SegmentSum", "SEGMENT_SUM"}, true);
    auto data = node.get_input(0);
    auto segment_ids = node.get_input(1);

    // create auxiliary constants
    auto const_one = create_same_type_const_scalar<int32_t>(segment_ids, 1);
    auto const_zero = create_same_type_const_scalar<int32_t>(segment_ids, 0);

    // compute SegmentSum using EmbeddingSegmentSum
    // for this prepare all the required inputs
    auto indices_type = segment_ids.get_element_type();
    // 1. compute a number of segments using segment_ids values
    // do not forget that segment ids are counting from zero
    auto reduction_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto num_segments_minus1 = make_shared<v1::ReduceMax>(segment_ids, reduction_axis, false);
    auto num_segments = make_shared<v1::Add>(num_segments_minus1, const_one);

    // 2. generate indices input for EmbeddingSegmentSum
    // that will collect slices consequently from data for each segment
    auto squeeze_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto segment_ids_shape = make_shared<v3::ShapeOf>(segment_ids, indices_type);
    auto num_indices = make_shared<v0::Squeeze>(segment_ids_shape, squeeze_axis);
    auto indices = make_shared<v4::Range>(const_zero, num_indices, const_one, indices_type);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        data = complex_type_mark->input_value(0);
        auto emb_segment_sum_complex = make_shared<v3::EmbeddingSegmentsSum>(data, indices, segment_ids, num_segments);
        auto emb_segment_sum_complex_output =
            make_shared<ComplexTypeMark>(emb_segment_sum_complex->output(0), complex_part_type);
        set_node_name(node.get_name(), emb_segment_sum_complex);
        return {emb_segment_sum_complex_output};
    }

    auto emb_segment_sum = make_shared<v3::EmbeddingSegmentsSum>(data, indices, segment_ids, num_segments);
    set_node_name(node.get_name(), emb_segment_sum);
    return {emb_segment_sum};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
