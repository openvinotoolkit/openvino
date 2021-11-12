// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <embedding_segments_sum_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, EmbeddingSegmentsSum) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = op::v0::Constant::create(element::i64, ov::Shape{}, {3});
    auto default_index = make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, Shape{4});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                         indices,
                                                         segment_ids,
                                                         num_segments,
                                                         default_index,
                                                         per_sample_weights);

    check_partial_shape(ess,
                        {ov::PartialShape{5, 2},
                         ov::PartialShape{4},
                         ov::PartialShape{4},
                         ov::PartialShape{},
                         ov::PartialShape{},
                         ov::PartialShape{4}},
                        {ov::PartialShape({3, 2})});

    check_static_shape(
        ess,
        {StaticShape{5, 2}, StaticShape{4}, StaticShape{4}, StaticShape{}, StaticShape{}, StaticShape{4}},
        {StaticShape{3, 2}});

    check_static_shape(ess,
                       {StaticShape{5, 2}, StaticShape{4}, StaticShape{4}, 8, StaticShape{}, StaticShape{4}},
                       {StaticShape{8, 2}});
}
