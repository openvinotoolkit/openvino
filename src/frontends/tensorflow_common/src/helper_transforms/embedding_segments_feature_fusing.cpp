// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/embedding_segments_feature_fusing.hpp"

#include <memory>
#include <vector>

#include "helper_ops/sparse_fill_empty_rows.hpp"
#include "helper_ops/sparse_segment_ops.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unique.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::op;

ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion::EmbeddingSegmentSingleFeatureFusion() {
    // The transformation looks for pattern (sub-graph) that performs extraction of embedding vectors from the
    // parameters table for object feature values, and sum up these embedding vectors for every object or compute their
    // mean value. Such sub-graph is met in the Wide and Deep model in case of the SINGLE categorical feature.
    auto embedding_table_pattern = ov::pass::pattern::any_input();
    auto input_values_pattern = ov::pass::pattern::any_input();
    auto input_indices_pattern = ov::pass::pattern::any_input();
    auto dense_shape_pattern = ov::pass::pattern::any_input();
    auto default_value_pattern = ov::pass::pattern::any_input();

    auto greaterequal0_const = make_shared<v0::Constant>(element::i64, Shape{}, vector<int64_t>{0});
    auto greaterequal0 = std::make_shared<v1::GreaterEqual>(input_values_pattern, greaterequal0_const);
    auto where0 = make_shared<v1::Transpose>(make_shared<v3::NonZero>(greaterequal0),
                                             make_shared<v0::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 0}));

    auto reshape0_shape = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    auto reshape0 = make_shared<v1::Reshape>(where0, reshape0_shape, false);
    auto gather0_1 = make_shared<v8::Gather>(input_indices_pattern,
                                             reshape0,
                                             make_shared<v0::Constant>(element::i32, Shape{}, vector<int32_t>{0}));
    auto gather0_2 = make_shared<v8::Gather>(input_values_pattern,
                                             reshape0,
                                             make_shared<v0::Constant>(element::i32, Shape{}, vector<int32_t>{0}));

    // SparseFillEmptyRows outputs segment ids along with indices for each segment. Indices correspond to vectors from
    // embedding table if some segment ids are not specified, SparseFillEmptyRows generate default indice for this
    // segment
    auto sparse_fill_empty_rows = make_shared<SparseFillEmptyRows>(gather0_1->output(0),
                                                                   gather0_2->output(0),
                                                                   dense_shape_pattern->output(0),
                                                                   default_value_pattern->output(0));

    auto strided_slice =
        make_shared<v1::StridedSlice>(sparse_fill_empty_rows->output(0),
                                      make_shared<v0::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 0}),
                                      make_shared<v0::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1}),
                                      make_shared<v0::Constant>(element::i32, Shape{2}, vector<int32_t>{1, 1}),
                                      std::vector<int64_t>{1},
                                      std::vector<int64_t>{1});
    auto cast = make_shared<v0::Convert>(strided_slice, ov::element::i64);

    auto unique = make_shared<v10::Unique>(sparse_fill_empty_rows->output(1), false, ov::element::i32);
    auto gather = make_shared<v8::Gather>(embedding_table_pattern,
                                          unique->output(0),
                                          make_shared<v0::Constant>(element::i64, Shape{}, vector<int64_t>{0}));

    // SparseSegmentSum sums-up extracted embedding vectors by indices for each segment
    auto sparse_segment_op = make_shared<SparseSegmentSum>(gather->output(0), unique->output(2), cast->output(0));

    auto shape = make_shared<v3::ShapeOf>(sparse_segment_op, ov::element::i32);
    auto strided_slice_for_shape_begin = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    auto strided_slice_for_shape_end = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{2});
    auto strided_slice_for_shape_step = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    auto strided_slice_for_shape = make_shared<v1::StridedSlice>(shape,
                                                                 strided_slice_for_shape_begin,
                                                                 strided_slice_for_shape_end,
                                                                 strided_slice_for_shape_step,
                                                                 std::vector<int64_t>{0},
                                                                 std::vector<int64_t>{0},
                                                                 std::vector<int64_t>{},
                                                                 std::vector<int64_t>{1});
    auto pack = make_shared<v0::Concat>(
        OutputVector{
            make_shared<v0::Unsqueeze>(make_shared<v0::Constant>(element::i32, Shape{}, 1),
                                       make_shared<v0::Constant>(element::i64, Shape{}, 0)),
            make_shared<v0::Unsqueeze>(strided_slice_for_shape, make_shared<v0::Constant>(element::i64, Shape{}, 0))},
        0);

    auto reshape = make_shared<v1::Reshape>(sparse_fill_empty_rows->output(2),
                                            make_shared<v0::Constant>(element::i32, Shape{2}, vector<int32_t>{-1, 1}),
                                            false);
    auto tile = make_shared<v0::Tile>(reshape, pack);

    auto zero_int_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_int_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    Output<Node> shape_of = make_shared<v3::ShapeOf>(sparse_segment_op, element::i32);
    shape_of = make_shared<v0::Concat>(OutputVector{one_int_const, shape_of}, 0);

    Output<Node> zeros_like =
        make_shared<v3::Broadcast>(make_shared<v0::Constant>(ov::element::f32, Shape{1}, std::vector<int64_t>{0}),
                                   shape_of);
    zeros_like = make_shared<v0::Squeeze>(zeros_like, zero_int_const);

    // compute number of dimensions to unsqueeze the condition
    auto cond_rank = compute_subgraph_scalar_rank(tile, element::i32);
    auto x_rank = compute_subgraph_scalar_rank(zeros_like, element::i32);
    auto num_new_axes = make_shared<v1::Subtract>(x_rank, cond_rank);

    // generate a new shape for the condition
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto new_subshape = make_shared<v3::Broadcast>(const_one, num_new_axes);
    auto cond_shape = make_shared<v3::ShapeOf>(tile, element::i32);
    // use extra dimensions in the begin to avoid concatenation of empty tensors that is not supported by Concat
    // remove this workaround once 100671 is resolved
    auto const_1 = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto new_cond_shape = make_shared<v0::Concat>(OutputVector{const_1, cond_shape, new_subshape}, 0);

    // prepare the condition to have the same rank as operands `x` and `y`
    auto prep_cond = make_shared<v1::Reshape>(tile, new_cond_shape, false)->output(0);
    // squeeze prep_cond by one extra dimension specially added
    auto const_0 = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    prep_cond = make_shared<v0::Squeeze>(prep_cond, const_0);

    auto select_pattern = make_shared<v1::Select>(prep_cond, zeros_like, sparse_segment_op);

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto embedding_table = pattern_map.at(embedding_table_pattern);
        const auto input_values = pattern_map.at(input_values_pattern);
        const auto input_indices = pattern_map.at(input_indices_pattern);
        const auto dense_shape = pattern_map.at(dense_shape_pattern);
        const auto default_value = pattern_map.at(default_value_pattern);

        auto select = as_type_ptr<v1::Select>(pattern_map.at(select_pattern).get_node_shared_ptr());

        // prepare input of indices for EmbeddingSegment operation
        auto cast_indices = make_shared<v0::Convert>(input_values, ov::element::i32);

        // prepare input of segment indices for EmbeddingSegment operation
        auto split_for_indices_axis = make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
        auto split_for_indices = make_shared<v1::Split>(input_indices, split_for_indices_axis, 2);
        auto squeeze_for_indices =
            make_shared<v0::Squeeze>(split_for_indices->output(0),
                                     make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 1));
        auto cast_segment_ids = make_shared<v0::Convert>(squeeze_for_indices, ov::element::i32);

        // prepare input of a number of segments for EmbeddingSegment operation
        auto split_for_dense_shape_axis = make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
        auto split_for_dense_shape = make_shared<v1::Split>(dense_shape, split_for_dense_shape_axis, 2);
        auto squeeze_to_scalar_axis = make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 0);
        auto squeeze_to_scalar = make_shared<v0::Squeeze>(split_for_dense_shape, squeeze_to_scalar_axis);
        auto cast_num_segments = make_shared<v0::Convert>(squeeze_to_scalar, ov::element::i32);

        // prepare the default value for EmbeddingSegment operation
        auto cast_default_value = make_shared<v0::Convert>(default_value, ov::element::i32);

        // TODO : remove Cast nodes once we start to support EmbeddingSegmentSum(new version) with segment_ids,
        // indices, and num_segments of different integer type.
        // Because the real cases show that it is possible to have it in TensorFlow
        auto embedding_segments_op = make_shared<v3::EmbeddingSegmentsSum>(embedding_table,
                                                                           cast_indices,
                                                                           cast_segment_ids,
                                                                           cast_num_segments,
                                                                           cast_default_value);
        embedding_segments_op->set_friendly_name(select->get_friendly_name());
        ov::copy_runtime_info(select, embedding_segments_op);
        ov::replace_node(select, embedding_segments_op);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(select_pattern,
                                                "ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion");
    register_matcher(m, callback);
}
