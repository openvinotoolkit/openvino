// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

inline void setup_4d_shape(ov_shape_t* shape, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    int64_t dims[4] = {d0, d1, d2, d3};
    ov_shape_create(4, dims, shape);
}

TEST(ov_tensor, ov_tensor_create) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape;
    setup_4d_shape(&shape, 10, 20, 30, 40);
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
    ov_shape_free(&shape);
}

TEST(ov_tensor, ov_tensor_create_from_host_ptr) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape;
    setup_4d_shape(&shape, 1, 3, 4, 4);
    uint8_t host_ptr[1][3][4][4] = {{{{0}}}};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create_from_host_ptr(type, shape, &host_ptr, &tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
    ov_shape_free(&shape);
}

static size_t product(const std::vector<size_t>& dims) {
    if (dims.empty())
        return 0;
    return std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
}

inline size_t calculate_size(ov_shape_t shape) {
    std::vector<size_t> tmp_shape;
    std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
    return product(tmp_shape);
}

inline size_t calculate_byteSize(ov_shape_t shape, ov_element_type_e type) {
    return (calculate_size(shape) * GET_ELEMENT_TYPE_SIZE(type) + 7) >> 3;
}

class ov_tensor_create_test : public ::testing::TestWithParam<ov_element_type_e> {
protected:
    void SetUp() override {
        ov_element_type_e type = GetParam();
        setup_4d_shape(&shape, 10, 20, 30, 40);
        tensor = nullptr;
        OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
        EXPECT_NE(nullptr, tensor);
    }

    void TearDown() override {
        ov_shape_free(&shape);
        ov_tensor_free(tensor);
    }

public:
    ov_shape_t shape;
    ov_tensor_t* tensor;
};

INSTANTIATE_TEST_SUITE_P(ov_tensor,
                         ov_tensor_create_test,
                         ::testing::Values(ov_element_type_e::BOOLEAN,
                                           ov_element_type_e::BF16,
                                           ov_element_type_e::F16,
                                           ov_element_type_e::F32,
                                           ov_element_type_e::F64,
                                           ov_element_type_e::I4,
                                           ov_element_type_e::I8,
                                           ov_element_type_e::I16,
                                           ov_element_type_e::I32,
                                           ov_element_type_e::I64,
                                           ov_element_type_e::U1,
                                           ov_element_type_e::U4,
                                           ov_element_type_e::U8,
                                           ov_element_type_e::U16,
                                           ov_element_type_e::U32,
                                           ov_element_type_e::U64));

TEST_P(ov_tensor_create_test, get_tensor_element_type) {
    ov_element_type_e type = GetParam();
    ov_element_type_e type_res;
    OV_EXPECT_OK(ov_tensor_get_element_type(tensor, &type_res));
    EXPECT_EQ(type, type_res);
}

TEST_P(ov_tensor_create_test, get_tensor_size) {
    size_t size = calculate_size(shape);
    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_size(tensor, &size_res));
    EXPECT_EQ(size_res, size);
}

TEST_P(ov_tensor_create_test, get_tensor_byte_size) {
    void* data = nullptr;
    OV_EXPECT_OK(ov_tensor_data(tensor, &data));
    EXPECT_NE(nullptr, data);
}

TEST_P(ov_tensor_create_test, get_tensor_data) {
    ov_element_type_e type = GetParam();
    size_t size = calculate_byteSize(shape, type);
    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_byte_size(tensor, &size_res));
    EXPECT_EQ(size_res, size);
}

TEST_P(ov_tensor_create_test, get_tensor_shape) {
    ov_shape_t shape_res;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    EXPECT_EQ(shape.rank, shape_res.rank);
    EXPECT_EQ(shape.dims[0], shape_res.dims[0]);
    EXPECT_EQ(shape.dims[1], shape_res.dims[1]);
    EXPECT_EQ(shape.dims[2], shape_res.dims[2]);
    EXPECT_EQ(shape.dims[3], shape_res.dims[3]);
    ov_shape_free(&shape_res);
}

TEST_P(ov_tensor_create_test, set_tensor_shape) {
    ov_shape_t shape_update;
    setup_4d_shape(&shape_update, 16, 16, 16, 16);
    OV_EXPECT_OK(ov_tensor_set_shape(tensor, shape_update));
    ov_shape_t shape_res;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    EXPECT_EQ(shape_update.rank, shape_res.rank);
    EXPECT_EQ(shape_update.dims[0], shape_res.dims[0]);
    EXPECT_EQ(shape_update.dims[1], shape_res.dims[1]);
    EXPECT_EQ(shape_update.dims[2], shape_res.dims[2]);
    EXPECT_EQ(shape_update.dims[3], shape_res.dims[3]);

    ov_shape_free(&shape_update);
    ov_shape_free(&shape_res);
}
