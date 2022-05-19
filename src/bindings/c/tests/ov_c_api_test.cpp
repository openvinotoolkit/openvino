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

class ov_compiled_model :public::testing::TestWithParam<std::string>{};
INSTANTIATE_TEST_CASE_P(device_name, ov_compiled_model, ::testing::Values("CPU"));

class ov_infer_request :public::testing::TestWithParam<std::string>{};
INSTANTIATE_TEST_CASE_P(device_name, ov_infer_request, ::testing::Values("CPU"));

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

TEST_P(ov_compiled_model, get_runtime_model) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t *runtime_model = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_runtime_model(compiled_model, &runtime_model));
    EXPECT_NE(nullptr, runtime_model);

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_runtime_model_error_handling) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_model_t *runtime_model = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(nullptr, &runtime_model));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_runtime_model(compiled_model, nullptr));

    ov_model_free(runtime_model);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_inputs) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t *input_nodes = new ov_output_node_list_t;
    input_nodes->output_nodes = nullptr;
    input_nodes->num = 0;
    OV_EXPECT_OK(ov_compiled_model_get_inputs(compiled_model, input_nodes));
    EXPECT_NE(nullptr, input_nodes->output_nodes);
    EXPECT_NE(0, input_nodes->num);

    ov_output_nodes_free(input_nodes);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_inputs_error_handling) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t *input_nodes = new ov_output_node_list_t;
    input_nodes->output_nodes = nullptr;
    input_nodes->num = 0;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_inputs(nullptr, input_nodes));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_inputs(compiled_model, nullptr));

    ov_output_nodes_free(input_nodes);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_outputs) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t *output_nodes = new ov_output_node_list_t;
    output_nodes->output_nodes = nullptr;
    output_nodes->num = 0;
    OV_EXPECT_OK(ov_compiled_model_get_outputs(compiled_model, output_nodes));
    EXPECT_NE(nullptr, output_nodes->output_nodes);
    EXPECT_NE(0, output_nodes->num);

    ov_output_nodes_free(output_nodes);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_outputs_error_handling) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_output_node_list_t *output_nodes = new ov_output_node_list_t;
    output_nodes->output_nodes = nullptr;
    output_nodes->num = 0;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_outputs(nullptr, output_nodes));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_outputs(compiled_model, nullptr));

    ov_output_nodes_free(output_nodes);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, create_infer_request) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, create_infer_request_error_handling) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(nullptr, &infer_request));
    OV_EXPECT_NOT_OK(ov_compiled_model_create_infer_request(compiled_model, nullptr));

    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_property) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_property_value property_value;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, SUPPORTED_PROPERTIES, &property_value));
    EXPECT_NE("", property_value.value_s);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_compiled_model, get_property_error_handling) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_property_value property_value;
    OV_EXPECT_NOT_OK(ov_compiled_model_get_property(nullptr, SUPPORTED_PROPERTIES, &property_value));
    OV_EXPECT_NOT_OK(ov_compiled_model_get_property(compiled_model, SUPPORTED_PROPERTIES, nullptr));

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

//InferRequest
TEST_P(ov_infer_request, set_tensor) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    const ov_element_type_e type = F32;
    const ov_shape_t shape = {1, 3, 32, 32};
    OV_EXPECT_OK(ov_tensor_create(type, shape, &input_tensor));
    EXPECT_NE(nullptr, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, set_tensor_error_handling) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    const ov_element_type_e type = U8;
    const ov_shape_t shape = {1, 3, 32, 32};
    OV_EXPECT_OK(ov_tensor_create(type, shape, &input_tensor));
    EXPECT_NE(nullptr, input_tensor);

    OV_EXPECT_NOT_OK(ov_infer_request_set_tensor(nullptr, "data", input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_set_tensor(infer_request, nullptr, input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_set_tensor(infer_request, "data", nullptr));

    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, get_tensor) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "data", &input_tensor));
    EXPECT_NE(nullptr, input_tensor);

    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, get_tensor_error_handing) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    OV_EXPECT_NOT_OK(ov_infer_request_get_tensor(nullptr, "data", &input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_get_tensor(infer_request, nullptr, &input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_get_tensor(infer_request, "data", nullptr));

    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, infer) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    const ov_element_type_e type = F32;
    const ov_shape_t shape = {1, 3, 32, 32};
    OV_EXPECT_OK(ov_tensor_create(type, shape, &input_tensor));
    EXPECT_NE(nullptr, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    OV_EXPECT_OK(ov_infer_request_infer(infer_request));

    ov_tensor_t *output_tensor = nullptr;
    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "fc_out", &output_tensor));
    EXPECT_NE(nullptr, output_tensor);

    ov_tensor_free(output_tensor);
    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, infer_ppp) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_preprocess_t *preprocess = nullptr;
    OV_EXPECT_OK(ov_preprocess_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t *preprocess_input_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_get_input_info_by_name(preprocess, "data", &preprocess_input_info));
    EXPECT_NE(nullptr, preprocess_input_info);

    ov_preprocess_input_tensor_info_t *preprocess_input_tensor_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_input_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info));
    EXPECT_NE(nullptr, preprocess_input_tensor_info);

    const ov_element_type_e element_type = U8;
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, element_type));
    
    OV_ASSERT_OK(ov_preprocess_build(preprocess, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    cv::Mat image = cv::imread(input_image);
    ov_tensor_t *input_tensor = nullptr;

    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "data", &input_tensor));
    EXPECT_NE(nullptr, input_tensor);
    mat_2_tensor(image, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    OV_EXPECT_OK(ov_infer_request_infer(infer_request));

    ov_tensor_t *output_tensor = nullptr;
    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "fc_out", &output_tensor));
    EXPECT_NE(nullptr, output_tensor);

    void *output_buffer = nullptr;
    OV_EXPECT_OK(ov_tensor_get_data(output_tensor, &output_buffer));
    EXPECT_NE(nullptr, output_buffer);
    float *output_data = (float*)output_buffer;
    EXPECT_NEAR(output_data[9], 0.f, 1.e-5);

    ov_tensor_free(output_tensor);
    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_preprocess_input_tensor_info_free(preprocess_input_tensor_info);
    ov_preprocess_input_info_free(preprocess_input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_infer_request, infer_error_handling) {
    OV_EXPECT_NOT_OK(ov_infer_request_infer(nullptr));
}

TEST_P(ov_infer_request, infer_async) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    const ov_element_type_e type = F32;
    const ov_shape_t shape = {1, 3, 32, 32};
    OV_EXPECT_OK(ov_tensor_create(type, shape, &input_tensor));
    EXPECT_NE(nullptr, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    OV_EXPECT_OK(ov_infer_request_start_async(infer_request));

    ov_tensor_t *output_tensor = nullptr;
    if (!HasFatalFailure()) {
        OV_EXPECT_OK(ov_infer_request_wait(infer_request));

        OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "fc_out", &output_tensor));
        EXPECT_NE(nullptr, output_tensor);
    }

    ov_tensor_free(output_tensor);
    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, infer_async_ppp) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_preprocess_t *preprocess = nullptr;
    OV_EXPECT_OK(ov_preprocess_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t *preprocess_input_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_get_input_info_by_name(preprocess, "data", &preprocess_input_info));
    EXPECT_NE(nullptr, preprocess_input_info);

    ov_preprocess_input_tensor_info_t *preprocess_input_tensor_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_input_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info));
    EXPECT_NE(nullptr, preprocess_input_tensor_info);

    const ov_element_type_e element_type = U8;
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, element_type));
    
    OV_ASSERT_OK(ov_preprocess_build(preprocess, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    cv::Mat image = cv::imread(input_image);
    ov_tensor_t *input_tensor = nullptr;

    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "data", &input_tensor));
    EXPECT_NE(nullptr, input_tensor);
    mat_2_tensor(image, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    OV_EXPECT_OK(ov_infer_request_start_async(infer_request));

    ov_tensor_t *output_tensor = nullptr;
    if (!HasFatalFailure()) {
        OV_EXPECT_OK(ov_infer_request_wait(infer_request));

        OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "fc_out", &output_tensor));
        EXPECT_NE(nullptr, output_tensor);

        void *output_buffer = nullptr;
        OV_EXPECT_OK(ov_tensor_get_data(output_tensor, &output_buffer));
        EXPECT_NE(nullptr, output_buffer);
        float *output_data = (float*)output_buffer;
        EXPECT_NEAR(output_data[9], 0.f, 1.e-5);
    }

    ov_tensor_free(output_tensor);
    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_preprocess_input_tensor_info_free(preprocess_input_tensor_info);
    ov_preprocess_input_info_free(preprocess_input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}

void infer_request_callback(void *args) {
    ov_infer_request_t *infer_request = (ov_infer_request_t *)args;
    ov_tensor_t *output_tensor = nullptr;

    printf("async infer callback...\n");
    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "fc_out", &output_tensor));
    EXPECT_NE(nullptr, output_tensor);

    ov_tensor_free(output_tensor);

    std::lock_guard<std::mutex> lock(m);
    ready = true;
    condVar.notify_one();
}

TEST_P(ov_infer_request, infer_request_set_callback) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ov_tensor_t *input_tensor = nullptr;
    const ov_element_type_e type = F32;
    const ov_shape_t shape = {1, 3, 32, 32};
    OV_EXPECT_OK(ov_tensor_create(type, shape, &input_tensor));
    EXPECT_NE(nullptr, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    ov_call_back_t callback;
    callback.callback_func = infer_request_callback;
    callback.args = infer_request;

    OV_EXPECT_OK(ov_infer_request_set_callback(infer_request, &callback));

    OV_EXPECT_OK(ov_infer_request_start_async(infer_request));

    if (!HasFatalFailure()) {
        std::unique_lock<std::mutex> lock(m);
        condVar.wait(lock, []{ return ready; });
    }

    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_infer_request, get_profiling_info) {
    auto device_name = GetParam();
    ov_core_t *core = nullptr;
    OV_ASSERT_OK(ov_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ov_model_t *model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_preprocess_t *preprocess = nullptr;
    OV_EXPECT_OK(ov_preprocess_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t *preprocess_input_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_get_input_info_by_name(preprocess, "data", &preprocess_input_info));
    EXPECT_NE(nullptr, preprocess_input_info);

    ov_preprocess_input_tensor_info_t *preprocess_input_tensor_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_input_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info));
    EXPECT_NE(nullptr, preprocess_input_tensor_info);

    const ov_element_type_e element_type = U8;
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, element_type));
    
    OV_ASSERT_OK(ov_preprocess_build(preprocess, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t *compiled_model = nullptr;
    ov_property_t property = {};
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), &compiled_model, &property));
    EXPECT_NE(nullptr, compiled_model);

    ov_infer_request_t *infer_request = nullptr;
    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    cv::Mat image = cv::imread(input_image);
    ov_tensor_t *input_tensor = nullptr;

    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "data", &input_tensor));
    EXPECT_NE(nullptr, input_tensor);
    mat_2_tensor(image, input_tensor);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, "data", input_tensor));

    OV_EXPECT_OK(ov_infer_request_infer(infer_request));

    ov_tensor_t *output_tensor = nullptr;
    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, "fc_out", &output_tensor));
    EXPECT_NE(nullptr, output_tensor);

    void *output_buffer = nullptr;
    OV_EXPECT_OK(ov_tensor_get_data(output_tensor, &output_buffer));
    EXPECT_NE(nullptr, output_buffer);
    float *output_data = (float*)output_buffer;
    EXPECT_NEAR(output_data[9], 0.f, 1.e-5);

    ov_profiling_info_list_t profiling_infos;
    profiling_infos.num = 0;
    profiling_infos.profiling_infos = nullptr;
    OV_EXPECT_OK(ov_infer_request_get_profiling_info(infer_request, &profiling_infos));
    EXPECT_NE(0, profiling_infos.num);
    EXPECT_NE(nullptr, profiling_infos.profiling_infos);

    std::cout << "profiling infor: " << profiling_infos.num << std::endl;
    for (int i = 0; i < profiling_infos.num; i++) {
        std::cout << "i " << i << std::endl;
        std::cout << "status " << profiling_infos.profiling_infos[i].status << std::endl;
        std::cout << "real_time " << profiling_infos.profiling_infos[i].real_time << std::endl;
        std::cout << "cpu_time " << profiling_infos.profiling_infos[i].cpu_time << std::endl;
        std::cout << "node_name " << profiling_infos.profiling_infos[i].node_name << std::endl;
        std::cout << "exec_type " << profiling_infos.profiling_infos[i].exec_type << std::endl;
        std::cout << "node_type " << profiling_infos.profiling_infos[i].node_type << std::endl;
    }

    ov_profiling_info_list_free(&profiling_infos);
    ov_tensor_free(output_tensor);
    ov_tensor_free(input_tensor);
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_preprocess_input_tensor_info_free(preprocess_input_tensor_info);
    ov_preprocess_input_info_free(preprocess_input_info);
    ov_preprocess_free(preprocess);
    ov_model_free(model);
    ov_core_free(core);
}