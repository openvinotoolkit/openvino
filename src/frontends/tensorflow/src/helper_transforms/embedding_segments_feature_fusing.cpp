// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/embedding_segments_feature_fusing.hpp"

#include <memory>
#include <vector>

#include "helper_ops/sparse_fill_empty_rows.hpp"
#include "helper_ops/sparse_segment_ops.hpp"
#include "helper_ops/unique.hpp"
#include "ngraph/rt_info.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::opset8;

ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion::EmbeddingSegmentSingleFeatureFusion() {
    auto input_values = ov::pass::pattern::any_input();
    auto input_indices = ov::pass::pattern::any_input();
    auto input_dense_shape = ov::pass::pattern::any_input();
    auto input_default_value = ov::pass::pattern::any_input();

    auto greaterequal0 =
        std::make_shared<GreaterEqual>(input_values, make_shared<Constant>(element::i64, Shape{}, vector<int64_t>{0}));
    auto where0 = make_shared<Transpose>(make_shared<NonZero>(greaterequal0),
                                         make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{1, 0}));

    auto reshape0 =
        make_shared<Reshape>(where0, make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{-1}), false);
    auto gather0_1 =
        make_shared<Gather>(input_indices, reshape0, make_shared<Constant>(element::i32, Shape{}, vector<int32_t>{0}));
    auto gather0_2 =
        make_shared<Gather>(input_values, reshape0, make_shared<Constant>(element::i32, Shape{}, vector<int32_t>{0}));

    auto sparse_fill_empty_rows = make_shared<SparseFillEmptyRows>(gather0_1->output(0),
                                                                   gather0_2->output(0),
                                                                   input_dense_shape->output(0),
                                                                   input_default_value->output(0));

    auto strided_slice = make_shared<StridedSlice>(sparse_fill_empty_rows->output(0),
                                                   make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{0, 0}),
                                                   make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1}),
                                                   make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{1, 1}),
                                                   std::vector<int64_t>{1},
                                                   std::vector<int64_t>{1});
    auto cast = make_shared<Convert>(strided_slice, ov::element::i64);

    auto unique = make_shared<Unique>(sparse_fill_empty_rows->output(1), ov::element::i32);
    auto gather = make_shared<Gather>(ov::pass::pattern::any_input(),
                                      unique->output(0),
                                      make_shared<Constant>(element::i64, Shape{}, vector<int64_t>{0}));

    auto sparse_segment_op = make_shared<SparseSegmentSum>(gather->output(0), unique->output(1), cast->output(0));

    auto shape = make_shared<ShapeOf>(sparse_segment_op, ov::element::i32);
    auto strided_slice_for_shape =
        make_shared<StridedSlice>(shape,
                                  make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{1}),
                                  make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{2}),
                                  make_shared<Constant>(element::i32, Shape{1}, vector<int32_t>{1}),
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
    auto select = make_shared<Select>(tile, zeros_like, sparse_segment_op);

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        int a = 1;
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(select,
                                                "ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion");
    register_matcher(m, callback);
}
