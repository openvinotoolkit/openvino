// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset11;
using namespace testing;

class TypePropBucketizeV3Test : public TypePropOpTest<op::v3::Bucketize> {};

TEST_F(TypePropBucketizeV3Test, default_ctor) {
    auto data = make_shared<Parameter>(element::f32, Shape{2, 3, 2});
    auto buckets = make_shared<Parameter>(element::f32, Shape{4});

    auto bucketize = make_op();
    bucketize->set_arguments(OutputVector{data, buckets});
    bucketize->set_output_type(element::i64);
    bucketize->set_with_right_bound(true);
    bucketize->validate_and_infer_types();

    EXPECT_TRUE(bucketize->get_with_right_bound());
    EXPECT_EQ(bucketize->get_output_type(), element::i64);
    EXPECT_EQ(bucketize->get_input_size(), 2);
    EXPECT_EQ(bucketize->get_output_size(), 1);
    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), (PartialShape{2, 3, 2}));
}

TEST_F(TypePropBucketizeV3Test, simple_shape) {
    auto data = make_shared<Parameter>(element::f32, Shape{2, 3, 2});
    auto buckets = make_shared<Parameter>(element::f32, Shape{4});
    auto bucketize = make_op(data, buckets);

    EXPECT_TRUE(bucketize->get_with_right_bound());
    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), (PartialShape{2, 3, 2}));
}

TEST_F(TypePropBucketizeV3Test, output_type_i32) {
    auto data = make_shared<Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto buckets = make_shared<Parameter>(element::f32, Shape{5});
    auto bucketize = make_op(data, buckets, element::i32);

    ASSERT_EQ(bucketize->get_output_element_type(0), element::i32);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), (PartialShape{1, 2, 3, 4}));
}

TEST_F(TypePropBucketizeV3Test, output_type_right_bound) {
    auto data = make_shared<Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto buckets = make_shared<Parameter>(element::f32, Shape{5});
    auto bucketize = make_op(data, buckets, element::i32, false);

    ASSERT_EQ(bucketize->get_output_element_type(0), element::i32);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), (PartialShape{1, 2, 3, 4}));
}

TEST_F(TypePropBucketizeV3Test, dynamic_input) {
    auto data_shape = PartialShape::dynamic();
    auto data = make_shared<Parameter>(element::f16, data_shape);
    auto buckets = make_shared<Parameter>(element::f32, Shape{5});
    auto bucketize = make_op(data, buckets);

    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropBucketizeV3Test, dynamic_buckets) {
    auto data = make_shared<Parameter>(element::f16, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto bucketize = make_op(data, buckets);

    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), (PartialShape{4, Dimension::dynamic()}));
}

TEST_F(TypePropBucketizeV3Test, interval_dimensions) {
    auto data_shape = PartialShape{{10, 30}, {12, -1}, -1, {0, 30}};
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<Parameter>(element::f16, data_shape);
    auto buckets = make_shared<Parameter>(element::f32, PartialShape{{2, 4}});
    auto bucketize = make_op(data, buckets);

    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_EQ(bucketize->get_output_partial_shape(0), data_shape);
    EXPECT_THAT(get_shape_symbols(bucketize->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropBucketizeV3Test, invalid_data_element_type) {
    auto data = make_shared<Parameter>(element::boolean, Shape{1, 2, 3, 4});
    auto buckets = make_shared<Parameter>(element::f32, Shape{5});
    OV_EXPECT_THROW(auto bucketize = make_op(data, buckets, element::i32),
                    NodeValidationFailure,
                    HasSubstr("Data input type must be numeric"));
}

TEST_F(TypePropBucketizeV3Test, invalid_bucket_element_types) {
    auto data = make_shared<Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto buckets = make_shared<Parameter>(element::boolean, Shape{5});

    OV_EXPECT_THROW(auto bucketize = make_op(data, buckets, element::i32),
                    NodeValidationFailure,
                    HasSubstr("Buckets input type must be numeric"));
}

TEST_F(TypePropBucketizeV3Test, invalid_output_types) {
    vector<element::Type_t> output_types = {element::f64,
                                            element::f32,
                                            element::f16,
                                            element::bf16,
                                            element::i16,
                                            element::i8,
                                            element::u64,
                                            element::u32,
                                            element::u16,
                                            element::u8,
                                            element::boolean};
    auto data = make_shared<Parameter>(element::f32, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<Parameter>(element::f32, Shape{5});
    for (const auto& output_type : output_types) {
        OV_EXPECT_THROW(auto bucketize = make_op(data, buckets, output_type),
                        NodeValidationFailure,
                        HasSubstr("Output type must be i32 or i64"));
    }
}

TEST_F(TypePropBucketizeV3Test, invalid_buckets_dim) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<Parameter>(element::f16, Shape{5, 5});
    OV_EXPECT_THROW(auto bucketize = make_op(data, buckets),
                    NodeValidationFailure,
                    HasSubstr("Buckets input must be a 1D tensor"));
}
