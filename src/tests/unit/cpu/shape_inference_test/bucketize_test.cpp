// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <bucketize_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, BucketizeV3) {
    auto data = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1, -1});
    auto buckets = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);

    check_static_shape(bucketize.get(), {ov::StaticShape{2, 3, 2}, ov::StaticShape{4}}, {ov::StaticShape{2, 3, 2}});
}
