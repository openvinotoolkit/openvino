// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bucketize_shape_inference.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class BucketizeV3StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v3::Bucketize> {
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(BucketizeV3StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_output_type(element::i32);
    op->set_with_right_bound(false);

    input_shapes = StaticShapeVector{{3, 2, 7, 89}, {3}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 2, 7, 89}));
}

TEST_F(BucketizeV3StaticShapeInferenceTest, dynamic_rank_inputs) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto buckets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(data, buckets, element::i32);

    input_shapes = StaticShapeVector{{10, 12, 1}, {5}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 12, 1}));
}

TEST_F(BucketizeV3StaticShapeInferenceTest, static_rank_inputs) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{-1, -1});
    const auto buckets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    op = make_op(data, buckets);

    input_shapes = StaticShapeVector{{100, 11}, {1}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({100, 11}));
}

TEST_F(BucketizeV3StaticShapeInferenceTest, bucket_incorrect_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{-1, -1});
    const auto buckets = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    op = make_op(data, buckets, element::i32);

    input_shapes = StaticShapeVector{{100, 11}, {2, 1}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Buckets input must be a 1D tensor"));
}
