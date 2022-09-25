// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/embedding_segments_feature_fusing.hpp"

#include <memory>
#include <vector>

#include "helper_ops/sparse_fill_empty_rows.hpp"
#include "helper_ops/sparse_segment_ops.hpp"
#include "helper_ops/unique.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::opset8;

ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion::EmbeddingSegmentSingleFeatureFusion() {
    // The transformation looks for pattern (sub-graph) that performs extraction of embedding vectors from the
    // parameters table for object feature values, and sum up these embedding vectors for every object or compute their
    // mean value. Such sub-graph is met in the Wide and Deep model in case of the SINGLE categorical feature.
    auto embedding_table_pattern = ov::pass::pattern::any_input();
    auto input_values_pattern = ov::pass::pattern::any_input();
    auto input_indices_pattern = ov::pass::pattern::any_input();
    auto dense_shape_pattern = ov::pass::pattern::any_input();
    auto default_value_pattern = ov::pass::pattern::any_input();

    auto greaterequal0_const = make_shared<Constant>(element::i64, Shape{}, vector<int64_t>{0});
    auto greaterequal0 = std::make_shared<GreaterEqual>(input_values_pattern, greaterequal0_const);
    auto where0 = make_shared<Transpose>(make_shared<NonZero>(greaterequal0),
                                         make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{1, 0}));

    auto reshape0_shape = make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    auto reshape0 = make_shared<Reshape>(where0, reshape0_shape, false);
    auto gather0_1 = make_shared<Gather>(input_indices_pattern,
                                         reshape0,
                                         make_shared<Constant>(element::i32, Shape{}, vector<int32_t>{0}));
    auto gather0_2 = make_shared<Gather>(input_values_pattern,
                                         reshape0,
                                         make_shared<Constant>(element::i32, Shape{}, vector<int32_t>{0}));

    // SparseFillEmptyRows outputs segment ids along with indices for each segment. Indices correspond to vectors from
    // embedding table if some segment ids are not specified, SparseFillEmptyRows generate default indice for this
    // segment
    auto sparse_fill_empty_rows = make_shared<SparseFillEmptyRows>(gather0_1->output(0),
                                                                   gather0_2->output(0),
                                                                   dense_shape_pattern->output(0),
                                                                   default_value_pattern->output(0));

    auto strided_slice = make_shared<StridedSlice>(sparse_fill_empty_rows->output(0),
                                                   make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{0, 0}),
                                                   make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1}),
                                                   make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{1, 1}),
                                                   std::vector<int64_t>{1},
                                                   std::vector<int64_t>{1});
    auto cast = make_shared<Convert>(strided_slice, ov::element::i64);

    auto unique = make_shared<Unique>(sparse_fill_empty_rows->output(1), ov::element::i32);
    auto gather = make_shared<Gather>(embedding_table_pattern,
                                      unique->output(0),
                                      make_shared<Constant>(element::i64, Shape{}, vector<int64_t>{0}));

    // SparseSegmentSum sums-up extracted embedding vectors by indices for each segment
    auto sparse_segment_op = make_shared<SparseSegmentSum>(gather->output(0), unique->output(1), cast->output(0));

    auto shape = make_shared<ShapeOf>(sparse_segment_op, ov::element::i32);
    auto strided_slice_for_shape_begin = make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    auto strided_slice_for_shape_end = make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{2});
    auto strided_slice_for_shape_step = make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    auto strided_slice_for_shape = make_shared<StridedSlice>(shape,
                                                             strided_slice_for_shape_begin,
                                                             strided_slice_for_shape_end,
                                                             strided_slice_for_shape_step,
                                                             std::vector<int64_t>{0},
                                                             std::vector<int64_t>{0},
                                                             std::vector<int64_t>{},
                                                             std::vector<int64_t>{1});
    auto pack = make_shared<Concat>(
        OutputVector{make_shared<Unsqueeze>(make_shared<Constant>(element::i32, Shape{}, 1),
                                            make_shared<Constant>(element::i64, Shape{}, 0)),
                     make_shared<Unsqueeze>(strided_slice_for_shape, make_shared<Constant>(element::i64, Shape{}, 0))},
        0);

    auto reshape = make_shared<Reshape>(sparse_fill_empty_rows->output(2),
                                        make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{-1, 1}),
                                        false);
    auto tile = make_shared<Tile>(reshape, pack);

    auto zeros_like = make_shared<Broadcast>(make_shared<Constant>(ov::element::f32, Shape{1}, std::vector<int64_t>{0}),
                                             make_shared<ShapeOf>(sparse_segment_op));
    auto select_pattern = make_shared<Select>(tile, zeros_like, sparse_segment_op);

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto embedding_table = pattern_map.at(embedding_table_pattern);
        const auto input_values = pattern_map.at(input_values_pattern);
        const auto input_indices = pattern_map.at(input_indices_pattern);
        const auto dense_shape = pattern_map.at(dense_shape_pattern);
        const auto default_value = pattern_map.at(default_value_pattern);

        auto select = as_type_ptr<Select>(pattern_map.at(select_pattern).get_node_shared_ptr());

        // prepare input of indices for EmbeddingSegment operation
        auto cast_indices = make_shared<Convert>(input_values, ov::element::i32);

        // prepare input of segment indices for EmbeddingSegment operation
        auto split_for_indices_axis = make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto split_for_indices = make_shared<Split>(input_indices, split_for_indices_axis, 2);
        auto squeeze_for_indices = make_shared<Squeeze>(split_for_indices->output(0),
                                                        make_shared<Constant>(ov::element::i64, ov::Shape{1}, 1));
        auto cast_segment_ids = make_shared<Convert>(squeeze_for_indices, ov::element::i32);

        // prepare input of a number of segments for EmbeddingSegment operation
        auto split_for_dense_shape_axis = make_shared<Constant>(ov::element::i64, ov::Shape{}, 0);
        auto split_for_dense_shape = make_shared<Split>(dense_shape, split_for_dense_shape_axis, 2);
        auto squeeze_to_scalar_axis = make_shared<Constant>(ov::element::i64, ov::Shape{1}, 0);
        auto squeeze_to_scalar = make_shared<Squeeze>(split_for_dense_shape, squeeze_to_scalar_axis);
        auto cast_num_segments = make_shared<Convert>(squeeze_to_scalar, ov::element::i32);

        // prepare the default value for EmbeddingSegment operation
        auto cast_default_value = make_shared<Convert>(default_value, ov::element::i32);

        // TODO : remove Cast nodes once we start to support EmbeddingSegmentSum(new version) with segment_ids,
        // indices, and num_segments of different integer type.
        // Because the real cases show that it is possible to have it in TensorFlow
        auto embedding_segments_op = make_shared<EmbeddingSegmentsSum>(embedding_table,
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
