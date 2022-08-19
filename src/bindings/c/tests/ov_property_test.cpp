// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_property, ov_properties_create_test) {
    ov_properties_t* property = nullptr;
    OV_ASSERT_OK(ov_properties_create(&property));

    ov_properties_free(property);
}

TEST(ov_property, ov_properties_create_and_add) {
    ov_properties_t* property = nullptr;
    OV_ASSERT_OK(ov_properties_create(&property));

    ov_property_key_e key = ov_property_key_e::PERFORMANCE_HINT;
    ov_performance_mode_e mode = ov_performance_mode_e::THROUGHPUT;
    ov_property_value_t value_1;
    value_1.ptr = (void*)&mode;
    value_1.size = 1;
    value_1.type = ov_property_value_type_e::ENUM;
    OV_ASSERT_OK(ov_properties_add(property, key, &value_1));

    key = ov_property_key_e::CACHE_DIR;
    const char cache_dir[] = "./cache_dir";
    ov_property_value_t value_2;
    value_2.ptr = (void*)cache_dir;
    value_2.size = sizeof(cache_dir);
    value_2.type = ov_property_value_type_e::CHAR;
    OV_ASSERT_OK(ov_properties_add(property, key, &value_2));

    key = ov_property_key_e::INFERENCE_NUM_THREADS;
    ov_property_value_t value_3;
    int32_t num = 8;
    value_3.ptr = (void*)&num;
    value_3.size = 1;
    value_3.type = ov_property_value_type_e::INT32;
    OV_ASSERT_OK(ov_properties_add(property, key, &value_3));

    EXPECT_EQ(ov_properties_size(property), 3);
    ov_properties_free(property);
}
