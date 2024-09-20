// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unique.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_sparse_segment_mean_op(const NodeContext& node) {
    default_op_checks(node, 3, {"SparseSegmentMean"});
    auto data = node.get_input(0);
    auto indices = std::make_shared<v0::Convert>(node.get_input(1), element::i64);
    auto segment_ids = std::make_shared<v0::Convert>(node.get_input(2), element::i64);
    auto data_rank = std::make_shared<v3::ShapeOf>(std::make_shared<v3::ShapeOf>(node.get_input(0)));

    // get the last index from segment_ids
    auto segments_ids_size = std::make_shared<v3::ShapeOf>(segment_ids, element::i64);
    auto const_one = create_same_type_const<int32_t>(indices, vector<int32_t>{1}, Shape{1});
    auto const_zero = create_same_type_const<int32_t>(indices, vector<int32_t>{0}, Shape{1});
    auto last_idx = std::make_shared<v1::Subtract>(segments_ids_size, const_one);

    // segment_ids are always sorted, so the last index from segment_ids can be used to determine the number of output
    // segments.
    auto last_segment_idx = std::make_shared<v8::Gather>(segment_ids, last_idx, const_zero);
    auto n_segments = std::make_shared<v1::Add>(last_segment_idx, const_one);

    // get sums of sparse segments
    auto embedding_segments_sum =
        make_shared<v3::EmbeddingSegmentsSum>(data, indices, segment_ids, std::make_shared<v0::Squeeze>(n_segments));

    // get the sizes of each segment
    auto unique_segment_ids = make_shared<v10::Unique>(segment_ids, true, element::i64, element::i64);
    auto broadcast = make_shared<v3::Broadcast>(const_one, n_segments);
    auto divisors = make_shared<v3::ScatterUpdate>(broadcast,
                                                   unique_segment_ids->output(0),
                                                   unique_segment_ids->output(3),
                                                   const_zero);
    auto divisors_with_correct_type = make_shared<v1::ConvertLike>(divisors, data);
    auto divisors_shape = make_shared<v3::ScatterUpdate>(make_shared<v3::Broadcast>(const_one, data_rank),
                                                         const_zero,
                                                         n_segments,
                                                         const_zero);
    auto divisors_with_correct_shape = std::make_shared<v1::Reshape>(divisors_with_correct_type, divisors_shape, false);

    // divide the sums by the size of the segments
    auto mean = std::make_shared<v1::Divide>(embedding_segments_sum, divisors_with_correct_shape);

    set_node_name(node.get_name(), mean);
    return {mean};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
