// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_partial_shape, ov_partial_shape_create_static) {
    const char* str = "{1,20,300,40}";
    ov_partial_shape_t partial_shape;

    int64_t rank = 4;
    int64_t dims[4] = {1, 20, 300, 40};

    OV_ASSERT_OK(ov_partial_shape_create_static(rank, dims, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_partial_shape_free(&partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_create_dynamic_1) {
    const char* str = "{1,20..30,300,40..100}";
    ov_partial_shape_t partial_shape;

    int64_t rank = 4;
    ov_dimension_t dims[4] = {{1, 1}, {20, 30}, {300, 300}, {40, 100}};

    OV_ASSERT_OK(ov_partial_shape_create(rank, dims, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_partial_shape_free(&partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_create_dynamic_2) {
    const char* str = "{1,20,300,40..100}";
    ov_partial_shape_t partial_shape;

    int64_t rank = 4;
    ov_dimension_t dims[4] = {{1, 1}, {20, 20}, {300, 300}, {40, 100}};

    OV_ASSERT_OK(ov_partial_shape_create(rank, dims, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_partial_shape_free(&partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_create_dynamic_3) {
    const char* str = "{1,?,300,40..100}";
    ov_partial_shape_t partial_shape;

    int64_t rank = 4;
    ov_dimension_t dims[4] = {{1, 1}, {-1, -1}, {300, 300}, {40, 100}};

    OV_ASSERT_OK(ov_partial_shape_create(rank, dims, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_partial_shape_free(&partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_create_dynamic_rank) {
    const char* str = "?";
    ov_partial_shape_t partial_shape;
    ov_rank_t rank = {-1, -1};

    OV_ASSERT_OK(ov_partial_shape_create_dynamic(rank, nullptr, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_partial_shape_free(&partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_create_invalid_dimension) {
    ov_partial_shape_t partial_shape = {{0, 0}, nullptr};

    int64_t rank = 4;
    ov_dimension_t dims[4] = {{1, 1}, {-1, -1}, {300, 100}, {40, 100}};
    OV_EXPECT_NOT_OK(ov_partial_shape_create(rank, dims, &partial_shape));
    ov_partial_shape_free(&partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_to_shape) {
    ov_partial_shape_t partial_shape;

    int64_t rank = 5;
    ov_dimension_t dims[5] = {{10, 10}, {20, 20}, {30, 30}, {40, 40}, {50, 50}};
    OV_ASSERT_OK(ov_partial_shape_create(rank, dims, &partial_shape));

    ov_shape_t shape;
    OV_ASSERT_OK(ov_partial_shape_to_shape(partial_shape, &shape));
    EXPECT_EQ(shape.rank, 5);
    EXPECT_EQ(shape.dims[0], 10);
    EXPECT_EQ(shape.dims[1], 20);
    EXPECT_EQ(shape.dims[2], 30);
    EXPECT_EQ(shape.dims[3], 40);
    EXPECT_EQ(shape.dims[4], 50);

    ov_partial_shape_free(&partial_shape);
    ov_shape_free(&shape);
}

TEST(ov_partial_shape, ov_partial_shape_to_shape_invalid) {
    ov_partial_shape_t partial_shape;

    int64_t rank = 5;
    ov_dimension_t dims[5] = {{10, 10}, {-1, -1}, {30, 30}, {40, 40}, {50, 50}};
    OV_ASSERT_OK(ov_partial_shape_create(rank, dims, &partial_shape));

    ov_shape_t shape;
    shape.rank = 0;
    OV_EXPECT_NOT_OK(ov_partial_shape_to_shape(partial_shape, &shape));

    ov_partial_shape_free(&partial_shape);
    ov_shape_free(&shape);
}

TEST(ov_partial_shape, ov_shape_to_partial_shape) {
    const char* str = "{10,20,30,40,50}";
    ov_shape_t shape;
    int64_t dims[5] = {10, 20, 30, 40, 50};
    OV_ASSERT_OK(ov_shape_create(5, dims, &shape));

    ov_partial_shape_t partial_shape;
    OV_ASSERT_OK(ov_shape_to_partial_shape(shape, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);

    EXPECT_STREQ(tmp, str);
    ov_partial_shape_free(&partial_shape);
    ov_shape_free(&shape);
    ov_free(tmp);
}

TEST(ov_partial_shape, ov_partial_shape_is_dynamic) {
    ov_partial_shape_t partial_shape;

    int64_t rank = 5;
    ov_dimension_t dims[5] = {{10, 10}, {-1, -1}, {30, 30}, {40, 40}, {50, 50}};
    OV_ASSERT_OK(ov_partial_shape_create(rank, dims, &partial_shape));
    ASSERT_TRUE(ov_partial_shape_is_dynamic(partial_shape));

    ov_partial_shape_t partial_shape_rank_dynamic;
    ov_rank_t rank_dynamic = {-1, -1};
    OV_ASSERT_OK(ov_partial_shape_create_dynamic(rank_dynamic, nullptr, &partial_shape_rank_dynamic));
    ASSERT_TRUE(ov_partial_shape_is_dynamic(partial_shape_rank_dynamic));

    ov_partial_shape_t partial_shape_static;
    int64_t rank_static = 5;
    ov_dimension_t dims_static[5] = {{10, 10}, {20, 20}, {30, 30}, {40, 40}, {50, 50}};
    OV_ASSERT_OK(ov_partial_shape_create(rank_static, dims_static, &partial_shape_static));
    ASSERT_FALSE(ov_partial_shape_is_dynamic(partial_shape_static));

    ov_partial_shape_free(&partial_shape);
    ov_partial_shape_free(&partial_shape_rank_dynamic);
    ov_partial_shape_free(&partial_shape_static);
}
