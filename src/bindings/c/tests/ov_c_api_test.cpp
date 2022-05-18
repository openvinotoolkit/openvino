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

class ov_core :public::testing::TestWithParam<std::string>{};
INSTANTIATE_TEST_CASE_P(device_name, ov_core, ::testing::Values("CPU"));

TEST(ov_core, ov_core_create_with_config) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(plugins_xml, &core));
    ASSERT_NE(nullptr, core);
    ov_core_free(core);
}

TEST(ov_core, ov_core_create_with_no_config) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model_no_bin) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

static std::vector<uint8_t> content_from_file(const char * filename, bool is_binary) {
    std::vector<uint8_t> result;
    {
        std::ifstream is(filename, is_binary ? std::ifstream::binary | std::ifstream::in : std::ifstream::in);
        if (is) {
            is.seekg(0, std::ifstream::end);
            result.resize(is.tellg());
            if (result.size() > 0) {
                is.seekg(0, std::ifstream::beg);
                is.read(reinterpret_cast<char *>(&result[0]), result.size());
            }
        }
    }
    return result;
}

TEST(ov_core, ov_core_read_model_from_memory) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    std::vector<uint8_t> weights_content(content_from_file(bin, true));

    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape = {1, weights_content.size()};
    OV_ASSERT_OK(ov_tensor_create_from_host_ptr(ov_element_type_e::U8, shape, weights_content.data(), &tensor));
    ASSERT_NE(nullptr, tensor);

    std::vector<uint8_t> xml_content(content_from_file(xml, false));
    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model_from_memory(core, reinterpret_cast<const char *>(xml_content.data()), tensor, &model));
    ASSERT_NE(nullptr, model);

    ov_tensor_free(tensor);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    ov_property_t property = {};
    OV_ASSERT_OK(ov_core_compile_model(core, model, devece_name.c_str(), &compiled_model, &property));
    ASSERT_NE(nullptr, compiled_model);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model_from_file) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    ov_property_t property = {};
    OV_ASSERT_OK(ov_core_compile_model_from_file(core, xml, devece_name.c_str(), &compiled_model, &property));
    ASSERT_NE(nullptr, compiled_model);

    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_property) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_property_t property{ov_property_key_e::PERFORMANCE_HINT, ov_performance_mode_e::THROUGHPUT, nullptr};
    OV_ASSERT_OK(ov_core_set_property(core, devece_name.c_str(), &property));
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_get_property) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_property_value property_value;
    OV_ASSERT_OK(ov_core_get_property(core, devece_name.c_str(), ov_property_key_e::SUPPORTED_PROPERTIES, &property_value));
    ov_core_free(core);
}

TEST(ov_core, ov_core_get_available_devices) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_available_devices_t devices;
    OV_ASSERT_OK(ov_core_get_available_devices(core, &devices));

    ov_available_devices_free(&devices);
    ov_core_free(core);
}

TEST_P(ov_core, ov_compiled_model_export) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    ov_property_t property = {};
    OV_ASSERT_OK(ov_core_compile_model_from_file(core, xml, devece_name.c_str(), &compiled_model, &property));
    ASSERT_NE(nullptr, compiled_model);

    std::string export_path = TestDataHelpers::generate_model_path("test_model", "exported_model.blob");
    OV_ASSERT_OK(ov_compiled_model_export(compiled_model, export_path.c_str()));
    
    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_import_model) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    ov_property_t property = {};
    OV_ASSERT_OK(ov_core_compile_model_from_file(core, xml, devece_name.c_str(), &compiled_model, &property));
    ASSERT_NE(nullptr, compiled_model);

    std::string export_path = TestDataHelpers::generate_model_path("test_model", "exported_model.blob");
    OV_ASSERT_OK(ov_compiled_model_export(compiled_model, export_path.c_str()));
    ov_compiled_model_free(compiled_model);

    std::vector<uchar> buffer(content_from_file(export_path.c_str(), true));
    ov_compiled_model_t* compiled_model_imported = nullptr;
    OV_ASSERT_OK(ov_core_import_model(core, reinterpret_cast<const char *>(buffer.data()), buffer.size(), devece_name.c_str(), &compiled_model_imported));
    ASSERT_NE(nullptr, compiled_model_imported);
    ov_compiled_model_free(compiled_model_imported);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_get_versions) {
    auto devece_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);
    
    ov_core_version_list_t version_list;
    OV_ASSERT_OK(ov_core_get_versions(core, devece_name.c_str(), &version_list));
    EXPECT_EQ(version_list.num_vers, 1);

    ov_core_versions_free(&version_list);
    ov_core_free(core);
}