// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <mutex>
#include "test_model_repo.hpp"
#include <fstream>

#include "c_api/ov_c_api.h"
#include "openvino/openvino.hpp"

std::string xml_std = TestDataHelpers::generate_model_path("test_model", "test_model_fp32.xml"),
            bin_std = TestDataHelpers::generate_model_path("test_model", "test_model_fp32.bin"),
            input_image_std = TestDataHelpers::generate_image_path("224x224", "dog.bmp"),
            input_image_nv12_std = TestDataHelpers::generate_image_path("224x224", "dog6.yuv");

const char* xml = xml_std.c_str();
const char* bin = bin_std.c_str();
const char* input_image = input_image_std.c_str();
const char* input_image_nv12 = input_image_nv12_std.c_str();

std::mutex m;
bool ready = false;
std::condition_variable condVar;
#ifdef _WIN32
    #ifdef __MINGW32__
        std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins_mingw.xml");
    #else
        std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins_win.xml");
    #endif
#elif defined __APPLE__
        std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins_apple.xml");
#else
        std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins.xml");
#endif
const char* plugins_xml = plugins_xml_std.c_str();

#define OV_EXPECT_OK(...) EXPECT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_ASSERT_OK(...) ASSERT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_NOT_OK(...) EXPECT_NE(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_ARREQ(arr1, arr2) EXPECT_TRUE(std::equal(std::begin(arr1), std::end(arr1), std::begin(arr2)))

size_t read_image_from_file(const char* img_path, unsigned char *img_data, size_t size) {
    FILE *fp = fopen(img_path, "rb+");
    size_t read_size = 0;

    if (fp) {
        fseek(fp, 0, SEEK_END);
        if (ftell(fp) >= size) {
            fseek(fp, 0, SEEK_SET);
            read_size = fread(img_data, 1, size, fp);
        }
        fclose(fp);
    }
    return read_size;
}

void mat_2_tensor(const cv::Mat& img, ov_tensor_t* tensor)
{
    ov_shape_t shape;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape));
    size_t channels = shape[1];
    size_t width = shape[3];
    size_t height = shape[2];
    void* tensor_data = NULL;
    OV_EXPECT_OK(ov_tensor_get_data(tensor, &tensor_data));
    uint8_t *tmp_data = (uint8_t *)(tensor_data);
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(width, height));

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                tmp_data[c * width * height + h * width + w] =
                        resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

size_t find_device(ov_available_devices_t avai_devices, const char *device_name) {
    for (size_t i = 0; i < avai_devices.num_devices; ++i) {
        if (strstr(avai_devices.devices[i], device_name))
            return i;
    }

    return -1;
}

TEST(ov_c_api_version, api_version) {
    ov_version_t version;
    ov_get_version(&version);
    auto ver = ov::get_openvino_version();
    std::string ver_str = ver.buildNumber;

    EXPECT_STREQ(version.buildNumber, ver.buildNumber);
    ov_version_free(&version);
}

TEST(ov_tensor_create, tensor_create) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {10, 20, 30, 40};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
}

TEST(ov_tensor_create_from_host_ptr, tensor_create_from_host_ptr) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {1, 3, 4, 4};
    uint8_t host_ptr[1][3][4][4]= {0};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create_from_host_ptr(type, shape, &host_ptr,&tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
}

TEST(ov_tensor_get_shape, tensor_get_shape) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape_t = {10, 20, 30, 40};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape_t, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_shape_t shape_res;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    OV_EXPECT_ARREQ(shape_t, shape_res);

    ov_tensor_free(tensor);
}

TEST(ov_tensor_set_shape, tensor_set_shape) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {1, 1, 1, 1};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_shape_t shape_t = {10, 20, 30, 40};
    OV_EXPECT_OK(ov_tensor_set_shape(tensor, shape_t));
    ov_shape_t shape_res;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    OV_EXPECT_ARREQ(shape_t, shape_res);

    ov_tensor_free(tensor);
}

TEST(ov_tensor_get_element_type, tensor_get_element_type) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape_t = {10, 20, 30, 40};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape_t, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_element_type_e type_res;
    OV_EXPECT_OK(ov_tensor_get_element_type(tensor, &type_res));
    EXPECT_EQ(type, type_res);

    ov_tensor_free(tensor);
}

TEST(ov_tensor_get_size, tensor_get_size) {
    ov_element_type_e type = ov_element_type_e::I16;
    ov_shape_t shape_t = {1, 3, 4, 4};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape_t, &tensor));
    EXPECT_NE(nullptr, tensor);

    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_size(tensor, &size_res));
    EXPECT_EQ(size_res, 48);

    ov_tensor_free(tensor);
}

TEST(ov_tensor_get_byte_size, tensor_get_byte_size) {
    ov_element_type_e type = ov_element_type_e::I16;
    ov_shape_t shape_t = {1, 3, 4, 4};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape_t, &tensor));
    EXPECT_NE(nullptr, tensor);

    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_byte_size(tensor, &size_res));
    EXPECT_EQ(size_res, 96);

    ov_tensor_free(tensor);
}

TEST(ov_tensor_get_data, tensor_get_data) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape_t = {10, 20, 30, 40};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape_t, &tensor));
    EXPECT_NE(nullptr, tensor);

    void *data = nullptr;
    OV_EXPECT_OK(ov_tensor_get_data(tensor, &data));
    EXPECT_NE(nullptr, data);

    ov_tensor_free(tensor);
}