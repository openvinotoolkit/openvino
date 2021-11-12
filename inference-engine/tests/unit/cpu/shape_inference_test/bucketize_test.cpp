// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <bucketize_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, BucketizeV3) {
    auto data = make_shared<op::v0::Parameter>(element::f32, ov::Shape{2, 3, 2});
    auto buckets = make_shared<op::v0::Parameter>(element::f32, ov::Shape{4});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);

    EXPECT_TRUE(bucketize->get_output_partial_shape(0).same_scheme(PartialShape{2, 3, 2}));

    check_partial_shape(bucketize, {ov::PartialShape{2, 3, 2}, ov::PartialShape{4}}, {ov::PartialShape{2, 3, 2}});

    check_static_shape(bucketize, {ov::StaticShape{2, 3, 2}, ov::StaticShape{4}}, {ov::StaticShape{2, 3, 2}});
}
