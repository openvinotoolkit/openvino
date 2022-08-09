// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <condition_variable>
#include <fstream>

#include "openvino/c/openvino.h"
#include "openvino/openvino.hpp"

extern const char* xml;
extern const char* bin;
extern const char* input_image;
extern const char* input_image_nv12;

extern const char* plugins_xml;

#define OV_EXPECT_OK(...)           EXPECT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_ASSERT_OK(...)           ASSERT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_NOT_OK(...)       EXPECT_NE(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_ARREQ(arr1, arr2) EXPECT_TRUE(std::equal(std::begin(arr1), std::end(arr1), std::begin(arr2)))

extern std::map<ov_element_type_e, size_t> element_type_size_map;
#define GET_ELEMENT_TYPE_SIZE(a) element_type_size_map[a]

inline size_t find_device(ov_available_devices_t avai_devices, const char* device_name) {
    for (size_t i = 0; i < avai_devices.size; ++i) {
        if (strstr(avai_devices.devices[i], device_name))
            return i;
    }

    return -1;
}

inline static std::vector<uint8_t> content_from_file(const char* filename, bool is_binary) {
    std::vector<uint8_t> result;
    {
        std::ifstream is(filename, is_binary ? std::ifstream::binary | std::ifstream::in : std::ifstream::in);
        if (is) {
            is.seekg(0, std::ifstream::end);
            result.resize(is.tellg());
            if (result.size() > 0) {
                is.seekg(0, std::ifstream::beg);
                is.read(reinterpret_cast<char*>(&result[0]), result.size());
            }
        }
    }
    return result;
}
