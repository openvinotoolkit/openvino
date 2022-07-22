// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <condition_variable>
#include <fstream>
#include <mutex>

#include "openvino/openvino.h"
#include "openvino/openvino.hpp"
#include "ov_test.hpp"

TEST(ov_tensor, ov_tensor_create) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {4, {10, 20, 30, 40}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_create_from_host_ptr) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {4, {1, 3, 4, 4}};
    uint8_t host_ptr[1][3][4][4] = {0};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create_from_host_ptr(type, shape, &host_ptr, &tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_get_shape) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {4, {10, 20, 30, 40}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_shape_t shape_res = {0, {0}};
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    EXPECT_EQ(shape.rank, shape_res.rank);
    OV_EXPECT_ARREQ(shape.dims, shape_res.dims);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_set_shape) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {4, {1, 1, 1, 1}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_shape_t shape_update = {4, {10, 20, 30, 40}};
    OV_EXPECT_OK(ov_tensor_set_shape(tensor, shape_update));
    ov_shape_t shape_res = {0, {0}};
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    EXPECT_EQ(shape_update.rank, shape_res.rank);
    OV_EXPECT_ARREQ(shape_update.dims, shape_res.dims);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_get_element_type) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {4, {10, 20, 30, 40}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_element_type_e type_res;
    OV_EXPECT_OK(ov_tensor_get_element_type(tensor, &type_res));
    EXPECT_EQ(type, type_res);

    ov_tensor_free(tensor);
}

static size_t product(const std::vector<size_t>& dims) {
    if (dims.empty())
        return 0;
    return std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
}

size_t calculate_size(ov_shape_t shape) {
    std::vector<size_t> tmp_shape;
    std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
    return product(tmp_shape);
}

size_t calculate_byteSize(ov_shape_t shape, ov_element_type_e type) {
    return (calculate_size(shape) * GET_ELEMENT_TYPE_SIZE(type) + 7) >> 3;
}

TEST(ov_tensor, ov_tensor_get_size) {
    ov_element_type_e type = ov_element_type_e::I16;
    ov_shape_t shape = {4, {1, 3, 4, 4}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    size_t size = calculate_size(shape);
    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_size(tensor, &size_res));
    EXPECT_EQ(size_res, size);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_get_byte_size) {
    ov_element_type_e type = ov_element_type_e::I16;
    ov_shape_t shape = {4, {1, 3, 4, 4}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    size_t size = calculate_byteSize(shape, type);
    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_byte_size(tensor, &size_res));
    EXPECT_EQ(size_res, size);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_data) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {4, {10, 20, 30, 40}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    void* data = nullptr;
    OV_EXPECT_OK(ov_tensor_data(tensor, &data));
    EXPECT_NE(nullptr, data);

    ov_tensor_free(tensor);
}