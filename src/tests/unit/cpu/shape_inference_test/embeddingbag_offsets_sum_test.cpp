// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <embeddingbag_offsets_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, EmbeddingBagOffsetsSumV3) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    auto offsets = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    auto default_index = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());

    auto ebos =
        make_shared<op::v3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index, per_sample_weights);

    check_static_shape(
        ebos.get(),
        {ov::StaticShape{5, 2}, ov::StaticShape{4}, ov::StaticShape{3}, ov::StaticShape{}, ov::StaticShape{4}},
        {ov::StaticShape{3, 2}});
}
