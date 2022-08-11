// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_partial_shape, ov_partial_shape_init_and_parse) {
    const char* str = "{1,20,300,40..100}";
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    ov_dimensions_t* dims = nullptr;
    OV_ASSERT_OK(ov_rank_create(&rank, 4));
    OV_ASSERT_OK(ov_dimensions_create(&dims));
    OV_ASSERT_OK(ov_dimensions_add(dims, 1));
    OV_ASSERT_OK(ov_dimensions_add(dims, 20));
    OV_ASSERT_OK(ov_dimensions_add(dims, 300));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, 40, 100));

    OV_ASSERT_OK(ov_partial_shape_create(&partial_shape, rank, dims));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_rank_free(rank);
    ov_dimensions_free(dims);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_init_and_parse_dynamic) {
    const char* str = "{1,?,300,40..100}";
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    ov_dimensions_t* dims = nullptr;
    OV_ASSERT_OK(ov_rank_create(&rank, 4));
    OV_ASSERT_OK(ov_dimensions_create(&dims));
    OV_ASSERT_OK(ov_dimensions_add(dims, 1));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, -1, -1));
    OV_ASSERT_OK(ov_dimensions_add(dims, 300));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, 40, 100));

    OV_ASSERT_OK(ov_partial_shape_create(&partial_shape, rank, dims));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_rank_free(rank);
    ov_dimensions_free(dims);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_init_and_parse_dynamic_mix) {
    const char* str = "{1,?,?,40..100}";
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    ov_dimensions_t* dims = nullptr;
    OV_ASSERT_OK(ov_rank_create(&rank, 4));
    OV_ASSERT_OK(ov_dimensions_create(&dims));
    OV_ASSERT_OK(ov_dimensions_add(dims, 1));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, -1, -1));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, -1, -1));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, 40, 100));

    OV_ASSERT_OK(ov_partial_shape_create(&partial_shape, rank, dims));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_rank_free(rank);
    ov_dimensions_free(dims);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_init_and_parse_dynamic_rank) {
    const char* str = "?";
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    OV_ASSERT_OK(ov_rank_create_dynamic(&rank, -1, -1));

    OV_ASSERT_OK(ov_partial_shape_create(&partial_shape, rank, nullptr));
    auto tmp = ov_partial_shape_to_string(partial_shape);
    EXPECT_STREQ(tmp, str);

    ov_free(tmp);
    ov_rank_free(rank);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_init_and_parse_invalid) {
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    ov_dimensions_t* dims = nullptr;
    OV_ASSERT_OK(ov_rank_create(&rank, 3));
    OV_ASSERT_OK(ov_dimensions_create(&dims));
    OV_ASSERT_OK(ov_dimensions_add(dims, 1));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, -1, -1));
    OV_ASSERT_OK(ov_dimensions_add(dims, 300));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, 40, 100));

    OV_EXPECT_NOT_OK(ov_partial_shape_create(&partial_shape, rank, dims));

    ov_rank_free(rank);
    ov_dimensions_free(dims);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_to_shape) {
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    ov_dimensions_t* dims = nullptr;
    OV_ASSERT_OK(ov_rank_create(&rank, 5));
    OV_ASSERT_OK(ov_dimensions_create(&dims));
    OV_ASSERT_OK(ov_dimensions_add(dims, 10));
    OV_ASSERT_OK(ov_dimensions_add(dims, 20));
    OV_ASSERT_OK(ov_dimensions_add(dims, 30));
    OV_ASSERT_OK(ov_dimensions_add(dims, 40));
    OV_ASSERT_OK(ov_dimensions_add(dims, 50));
    OV_EXPECT_OK(ov_partial_shape_create(&partial_shape, rank, dims));

    ov_shape_t shape;
    OV_ASSERT_OK(ov_partial_shape_to_shape(partial_shape, &shape));
    EXPECT_EQ(shape.rank, 5);
    EXPECT_EQ(shape.dims[0], 10);
    EXPECT_EQ(shape.dims[1], 20);
    EXPECT_EQ(shape.dims[2], 30);
    EXPECT_EQ(shape.dims[3], 40);
    EXPECT_EQ(shape.dims[4], 50);

    ov_rank_free(rank);
    ov_dimensions_free(dims);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_partial_shape_to_shape_invalid) {
    ov_partial_shape_t* partial_shape = nullptr;

    ov_rank_t* rank = nullptr;
    ov_dimensions_t* dims = nullptr;
    OV_ASSERT_OK(ov_rank_create(&rank, 5));
    OV_ASSERT_OK(ov_dimensions_create(&dims));
    OV_ASSERT_OK(ov_dimensions_add(dims, 10));
    OV_ASSERT_OK(ov_dimensions_add_dynamic(dims, -1, -1));
    OV_ASSERT_OK(ov_dimensions_add(dims, 30));
    OV_ASSERT_OK(ov_dimensions_add(dims, 40));
    OV_ASSERT_OK(ov_dimensions_add(dims, 50));

    ov_shape_t shape;
    shape.rank = 0;
    OV_EXPECT_NOT_OK(ov_partial_shape_to_shape(partial_shape, &shape));

    ov_rank_free(rank);
    ov_dimensions_free(dims);
    ov_partial_shape_free(partial_shape);
}

TEST(ov_partial_shape, ov_shape_to_partial_shape) {
    const char* str = "{10,20,30,40,50}";
    ov_shape_t shape;
    OV_ASSERT_OK(ov_shape_init(&shape, 5));
    shape.dims[0] = 10;
    shape.dims[1] = 20;
    shape.dims[2] = 30;
    shape.dims[3] = 40;
    shape.dims[4] = 50;

    ov_partial_shape_t* partial_shape = nullptr;
    OV_ASSERT_OK(ov_shape_to_partial_shape(&shape, &partial_shape));
    auto tmp = ov_partial_shape_to_string(partial_shape);

    EXPECT_STREQ(tmp, str);
    ov_partial_shape_free(partial_shape);
    ov_shape_deinit(&shape);
}
