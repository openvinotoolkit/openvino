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

std::map<ov_element_type_e, size_t> element_type_size_map = {
        {ov_element_type_e::BOOLEAN, 8},
        {ov_element_type_e::BF16, 16},
        {ov_element_type_e::F16, 16},
        {ov_element_type_e::F32, 32},
        {ov_element_type_e::F64, 64},
        {ov_element_type_e::I4, 4},
        {ov_element_type_e::I8, 8},
        {ov_element_type_e::I16, 16},
        {ov_element_type_e::I32, 32},
        {ov_element_type_e::I64, 64},
        {ov_element_type_e::U1, 1},
        {ov_element_type_e::U4, 4},
        {ov_element_type_e::U8, 8},
        {ov_element_type_e::U16, 16},
        {ov_element_type_e::U32, 32},
        {ov_element_type_e::U64, 64}};
#define GET_ELEMENT_TYPE_SIZE(a) element_type_size_map[a]

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

TEST(ov_preprocess, ov_preprocess_create) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_get_input_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info(preprocess, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_get_input_info_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_name(preprocess, "data", &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_get_input_info_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_get_tensor_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_get_preprocess_steps) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_process_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);

    ov_preprocess_input_process_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_resize) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_process_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);

    OV_ASSERT_OK(ov_preprocess_input_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));

    ov_preprocess_input_process_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_element_type) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::F32));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_tensor) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape = {1, 416, 416, 3};
    OV_ASSERT_OK(ov_tensor_create(ov_element_type_e::F32, shape, &tensor));
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_tensor(input_tensor_info, tensor));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_tensor_info_set_layout) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);

    ov_layout_t layout = {'N', 'C', 'H', 'W'};
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout));

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_get_output_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_output_info(preprocess, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_info_free(output_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_get_output_info_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_info_free(output_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_get_output_info_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_output_info_by_name(preprocess, "fc_out", &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_info_free(output_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_output_get_tensor_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_tensor_info_t* output_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_output_get_tensor_info(output_info, &output_tensor_info));
    ASSERT_NE(nullptr, output_tensor_info);

    ov_preprocess_output_tensor_info_free(output_tensor_info);
    ov_preprocess_output_info_free(output_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_output_set_element_type) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);

    ov_preprocess_output_tensor_info_t* output_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_output_get_tensor_info(output_info, &output_tensor_info));
    ASSERT_NE(nullptr, output_tensor_info);

    OV_ASSERT_OK(ov_preprocess_output_set_element_type(output_tensor_info, ov_element_type_e::F32));

    ov_preprocess_output_tensor_info_free(output_tensor_info);
    ov_preprocess_output_info_free(output_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_get_model_info) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_model_info_t* input_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_model_info(input_info, &input_model));
    ASSERT_NE(nullptr, input_model);

    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_input_model_set_layout) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_model_info_t* input_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_model_info(input_info, &input_model));
    ASSERT_NE(nullptr, input_model);

    ov_layout_t layout = {'N', 'C', 'H', 'W'};
    OV_ASSERT_OK(ov_preprocess_input_model_set_layout(input_model, layout));

    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_build) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_model_t* new_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_build(preprocess, &new_model));
    ASSERT_NE(nullptr, new_model);

    ov_model_free(new_model);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_preprocess, ov_preprocess_build_apply) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_preprocess_t* preprocess = nullptr;
    OV_ASSERT_OK(ov_preprocess_create(model, &preprocess));
    ASSERT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* input_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));
    ASSERT_NE(nullptr, input_info);

    ov_preprocess_input_tensor_info_t* input_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    ASSERT_NE(nullptr, input_tensor_info);
    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape = {1, 416, 416, 3};
    OV_ASSERT_OK(ov_tensor_create(ov_element_type_e::U8, shape, &tensor));
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_tensor(input_tensor_info, tensor));
    ov_layout_t tensor_layout = {'N', 'H', 'W', 'C'};
    OV_ASSERT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, tensor_layout));

    ov_preprocess_input_process_steps_t* input_process = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_preprocess_steps(input_info, &input_process));
    ASSERT_NE(nullptr, input_process);
    OV_ASSERT_OK(ov_preprocess_input_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));

    ov_preprocess_input_model_info_t* input_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_input_get_model_info(input_info, &input_model));
    ASSERT_NE(nullptr, input_model);
    ov_layout_t model_layout = {'N', 'C', 'H', 'W'};
    OV_ASSERT_OK(ov_preprocess_input_model_set_layout(input_model, model_layout));

    ov_preprocess_output_info_t* output_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_get_output_info_by_index(preprocess, 0, &output_info));
    ASSERT_NE(nullptr, output_info);
    ov_preprocess_output_tensor_info_t* output_tensor_info = nullptr;
    OV_ASSERT_OK(ov_preprocess_output_get_tensor_info(output_info, &output_tensor_info));
    ASSERT_NE(nullptr, output_tensor_info);
    OV_ASSERT_OK(ov_preprocess_output_set_element_type(output_tensor_info, ov_element_type_e::F32));

    ov_model_t* new_model = nullptr;
    OV_ASSERT_OK(ov_preprocess_build(preprocess, &new_model));
    ASSERT_NE(nullptr, new_model);

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_tensor_free(tensor);
    ov_preprocess_input_process_steps_free(input_process);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_output_tensor_info_free(output_tensor_info);
    ov_preprocess_output_info_free(output_info);
    ov_preprocess_input_info_free(input_info);
    ov_model_free(new_model);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_tensor, ov_tensor_create) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {10, 20, 30, 40};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_create_from_host_ptr) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {1, 3, 4, 4};
    uint8_t host_ptr[1][3][4][4]= {0};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create_from_host_ptr(type, shape, &host_ptr,&tensor));
    EXPECT_NE(nullptr, tensor);
    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_get_shape) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {10, 20, 30, 40};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_shape_t shape_res;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    OV_EXPECT_ARREQ(shape, shape_res);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_set_shape) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {1, 1, 1, 1};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    ov_shape_t shape_update = {10, 20, 30, 40};
    OV_EXPECT_OK(ov_tensor_set_shape(tensor, shape_update));
    ov_shape_t shape_res;
    OV_EXPECT_OK(ov_tensor_get_shape(tensor, &shape_res));
    OV_EXPECT_ARREQ(shape_update, shape_res);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_get_element_type) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {10, 20, 30, 40};
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
    std::copy_if(shape, shape + 4,
                 std::back_inserter(tmp_shape),
                 [](size_t x) { return x != 0;});
    return product(tmp_shape);
}

size_t calculate_byteSize(ov_shape_t shape, ov_element_type_e type) {
    return (calculate_size(shape) * GET_ELEMENT_TYPE_SIZE(type) + 7) >> 3;
}

TEST(ov_tensor, ov_tensor_get_size) {
    ov_element_type_e type = ov_element_type_e::I16;
    ov_shape_t shape = {1, 3, 4, 4};
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
    ov_shape_t shape = {1, 3, 4, 4};
    ov_tensor_t* tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    size_t size = calculate_byteSize(shape, type);
    size_t size_res;
    OV_EXPECT_OK(ov_tensor_get_byte_size(tensor, &size_res));
    EXPECT_EQ(size_res, size);

    ov_tensor_free(tensor);
}

TEST(ov_tensor, ov_tensor_get_data) {
    ov_element_type_e type = ov_element_type_e::U8;
    ov_shape_t shape = {10, 20, 30, 40};
    ov_tensor_t *tensor = nullptr;
    OV_EXPECT_OK(ov_tensor_create(type, shape, &tensor));
    EXPECT_NE(nullptr, tensor);

    void *data = nullptr;
    OV_EXPECT_OK(ov_tensor_get_data(tensor, &data));
    EXPECT_NE(nullptr, data);

    ov_tensor_free(tensor);
}

TEST(ov_model, ov_model_get_outputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t output_node_list;
    output_node_list.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_get_outputs(model, &output_node_list));
    ASSERT_NE(nullptr, output_node_list.output_nodes);

    ov_output_node_list_free(&output_node_list);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_get_inputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t input_node_list;
    input_node_list.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_get_inputs(model, &input_node_list));
    ASSERT_NE(nullptr, input_node_list.output_nodes);

    ov_output_node_list_free(&input_node_list);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_get_input_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* input_node = nullptr;
    OV_ASSERT_OK(ov_model_get_input_by_name(model, "data", &input_node));
    ASSERT_NE(nullptr, input_node);

    ov_output_node_free(input_node);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_get_input_by_id) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* input_node = nullptr;
    OV_ASSERT_OK(ov_model_get_input_by_id(model, 0, &input_node));
    ASSERT_NE(nullptr, input_node);

    ov_output_node_free(input_node);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_is_dynamic) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ASSERT_FALSE(ov_model_is_dynamic(model));

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_reshape) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t input_node_list1;
    input_node_list1.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_get_inputs(model, &input_node_list1));
    ASSERT_NE(nullptr, input_node_list1.output_nodes);

    ov_partial_shape_t partial_shape = {"1","3","64","?"};
    OV_ASSERT_OK(ov_model_reshape(model, "data", partial_shape));

    ov_output_node_list_t input_node_list2;
    input_node_list2.output_nodes = nullptr;
    OV_ASSERT_OK(ov_model_get_inputs(model, &input_node_list2));
    ASSERT_NE(nullptr, input_node_list2.output_nodes);

    EXPECT_NE(input_node_list1.output_nodes, input_node_list2.output_nodes);

    ov_output_node_list_free(&input_node_list1);
    ov_output_node_list_free(&input_node_list2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_get_friendly_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    char* friendly_name = nullptr;
    OV_ASSERT_OK(ov_model_get_friendly_name(model, &friendly_name));
    ASSERT_NE(nullptr, friendly_name);

    ov_model_free(model);
    ov_core_free(core);
}