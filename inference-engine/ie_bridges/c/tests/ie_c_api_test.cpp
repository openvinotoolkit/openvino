// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier : Apache-2.0
//

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <mutex>
#include <c_api/ie_c_api.h>
#include <inference_engine.hpp>
#include "test_model_repo.hpp"
#include <fstream>

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

#define IE_EXPECT_OK(...) EXPECT_EQ(IEStatusCode::OK, __VA_ARGS__)
#define IE_ASSERT_OK(...) ASSERT_EQ(IEStatusCode::OK, __VA_ARGS__)
#define IE_EXPECT_NOT_OK(...) EXPECT_NE(IEStatusCode::OK, __VA_ARGS__)

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

void Mat2Blob(const cv::Mat& img, ie_blob_t *blob)
{
    dimensions_t dimenison;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dimenison));
    size_t channels = dimenison.dims[1];
    size_t width = dimenison.dims[3];
    size_t height = dimenison.dims[2];
    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &buffer));
    uint8_t *blob_data = (uint8_t *)(buffer.buffer);
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(width, height));

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[c * width * height + h * width + w] =
                        resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

size_t find_device(ie_available_devices_t avai_devices, const char *device_name) {
    for (size_t i = 0; i < avai_devices.num_devices; ++i) {
        if (strstr(avai_devices.devices[i], device_name))
            return i;
    }

    return -1;
}

TEST(ie_c_api_version, apiVersion) {
    ie_version_t version = ie_c_api_version();
    auto ver = InferenceEngine::GetInferenceEngineVersion();
    std::string ver_str = std::to_string(ver->apiVersion.major) + ".";
    ver_str += std::to_string(ver->apiVersion.minor) + ".";
    ver_str += ver->buildNumber;

    EXPECT_EQ(strcmp(version.api_version, ver_str.c_str()), 0);
    ie_version_free(&version);
}

void completion_callback(void *args) {
    ie_infer_request_t *infer_request = (ie_infer_request_t *)args;
    ie_blob_t *output_blob = nullptr;

    printf("async infer callback...\n");
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "fc_out", &output_blob));

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
    float *output_data = (float *)(buffer.buffer);
    EXPECT_NEAR(output_data[9], 0.f, 1.e-5);

    ie_blob_free(&output_blob);

    std::lock_guard<std::mutex> lock(m);
    ready = true;
    condVar.notify_one();
}

TEST(ie_core_create, coreCreatewithConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create(plugins_xml, &core));
    ASSERT_NE(nullptr, core);

    ie_core_free(&core);
}

TEST(ie_core_create, coreCreateNoConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_core_free(&core);
}

TEST(ie_core_get_available_devices, getAvailableDevices) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));

    ie_available_devices_t avai_devices = {0};
    IE_EXPECT_OK(ie_core_get_available_devices(core, &avai_devices));

    ie_core_available_devices_free(&avai_devices);
    ie_core_free(&core);
}

TEST(ie_core_register_plugin, registerPlugin) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *plugin_name = "MKLDNNPlugin";
    const char *device_name = "BLA";
    IE_EXPECT_OK(ie_core_register_plugin(core, plugin_name, device_name));

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_core_register_plugins, registerPlugins) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_core_register_plugins(core, plugins_xml));

    ie_config_t config = {nullptr, nullptr, nullptr};
    const char *device_name = "CUSTOM";
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_core_unregister_plugin, unregisterPlugin) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create(plugins_xml, &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    ie_config_t config = {nullptr, nullptr, nullptr};
    const char *device_name = "CUSTOM";
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);

    IE_EXPECT_OK(ie_core_unregister_plugin(core, device_name));

    ie_core_free(&core);
}

TEST(ie_core_set_config, setConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    ie_config_t config = {"CPU_THREADS_NUM", "3", nullptr};
    IE_EXPECT_OK(ie_core_set_config(core, &config, device_name));

    ie_core_free(&core);
}

TEST(ie_core_get_metric, getMetric) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    const char *metric_name = "SUPPORTED_CONFIG_KEYS";
    ie_param_t param;
    param.params = nullptr;
    IE_EXPECT_OK(ie_core_get_metric(core, device_name, metric_name, &param));

    ie_param_free(&param);
    ie_core_free(&core);
}

TEST(ie_core_get_config, getConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    const char *config_name = "CPU_THREADS_NUM";
    ie_param_t param;
    param.params = nullptr;
    IE_EXPECT_OK(ie_core_get_config(core, device_name, config_name, &param));
    EXPECT_STREQ(param.params, "0");

    ie_param_free(&param);
    ie_core_free(&core);
}

TEST(ie_core_get_versions, getVersions) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_core_versions_t versions = {0};
    IE_EXPECT_OK(ie_core_get_versions(core, "CPU", &versions));
    EXPECT_EQ(versions.num_vers, 1);

    ie_core_versions_free(&versions);
    ie_core_free(&core);
}

TEST(ie_core_read_network, networkRead) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    ie_network_free(&network);
    ie_core_free(&core);
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

TEST(ie_core_read_network_from_memory, networkReadFromMemory) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    std::vector<uint8_t> weights_content(content_from_file(bin, true));

    tensor_desc_t weights_desc { ANY, { 1, { weights_content.size() } }, U8 };
    ie_blob_t *weights_blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&weights_desc, weights_content.data(), weights_content.size(), &weights_blob));
    EXPECT_NE(nullptr, weights_blob);

    if (weights_blob != nullptr) {
        std::vector<uint8_t> xml_content(content_from_file(xml, false));
        
        ie_network_t *network = nullptr;
        IE_EXPECT_OK(ie_core_read_network_from_memory(core, xml_content.data(), xml_content.size(), weights_blob, &network));
        EXPECT_NE(nullptr, network);
        if (network != nullptr) {
            ie_network_free(&network);
        }
        ie_blob_free(&weights_blob);
    }

    ie_core_free(&core);
}

TEST(ie_core_load_network, loadNetwork) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_layout(network, "data", layout_e::NHWC));
    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));

    ie_config_t config = {"CPU_THREADS_NUM", "3", nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_core_load_network, loadNetworkNoConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_name, networkName) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    char *network_name = nullptr;
    IE_EXPECT_OK(ie_network_get_name(network, &network_name));

    EXPECT_STREQ(network_name, "test_model");

    ie_network_name_free(&network_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_inputs_number, inputNumer) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    size_t size;
    IEStatusCode status_result = ie_network_get_inputs_number(network, &size);
    EXPECT_EQ(status_result, IEStatusCode::OK);
    EXPECT_EQ(size, 1);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_name, inputName) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    char *input_name = nullptr;
    IE_EXPECT_OK(ie_network_get_input_name(network, 0, &input_name));

    EXPECT_STREQ(input_name, "data");

    ie_network_name_free(&input_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_precision, getPrecision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    precision_e p;
    IE_EXPECT_OK(ie_network_get_input_precision(network, name, &p));
    EXPECT_EQ(p, precision_e::FP32);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_precision, incorrectName) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "model";
    precision_e p;
    EXPECT_EQ(IEStatusCode::NOT_FOUND, ie_network_get_input_precision(network, name, &p));

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_set_input_precision, setPrecision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    const precision_e p = precision_e::FP16;
    IE_EXPECT_OK(ie_network_set_input_precision(network, name, p));
    precision_e p2;
    IE_EXPECT_OK(ie_network_get_input_precision(network, name, &p2));
    EXPECT_EQ(p, p2);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_layout, getLayout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    layout_e l;
    IE_EXPECT_OK(ie_network_get_input_layout(network, name, &l));
    EXPECT_EQ(l, layout_e::NCHW);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_set_input_layout, setLayout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    const layout_e l = layout_e ::NHWC;
    IE_EXPECT_OK(ie_network_set_input_layout(network, name, l));
    layout_e l2;
    IE_EXPECT_OK(ie_network_get_input_layout(network, name, &l2));
    EXPECT_EQ(l, l2);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_dims, getDims) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    dimensions_t dims_res;
    IE_EXPECT_OK(ie_network_get_input_dims(network, name, &dims_res));
    EXPECT_EQ(dims_res.dims[0], 1);
    EXPECT_EQ(dims_res.dims[1], 3);
    EXPECT_EQ(dims_res.dims[2], 32);
    EXPECT_EQ(dims_res.dims[3], 32);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_resize_algorithm, getResizeAlgo) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    resize_alg_e resizeAlg;
    IE_EXPECT_OK(ie_network_get_input_resize_algorithm(network, name, &resizeAlg));
    EXPECT_EQ(resizeAlg, resize_alg_e::NO_RESIZE);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_set_input_resize_algorithm, setResizeAlgo) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    resize_alg_e resizeAlg = resize_alg_e::RESIZE_BILINEAR;
    IE_EXPECT_OK(ie_network_set_input_resize_algorithm(network, name, resizeAlg));

    resize_alg_e resizeAlg2;
    IE_EXPECT_OK(ie_network_get_input_resize_algorithm(network, name, &resizeAlg2));
    EXPECT_EQ(resizeAlg, resizeAlg2);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_color_format, getColorFormat) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    colorformat_e color;
    IE_EXPECT_OK(ie_network_get_color_format(network, name, &color));
    EXPECT_EQ(color, colorformat_e::RAW);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_set_color_format, setColorFormat) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "data";
    const colorformat_e color = colorformat_e::BGR;
    IE_EXPECT_OK(ie_network_set_color_format(network, name, color));

    colorformat_e color2;
    IE_EXPECT_OK(ie_network_get_color_format(network, name, &color2));
    EXPECT_EQ(color2, colorformat_e::BGR);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_input_shapes, getInputShapes) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    input_shapes_t shapes;
    IE_EXPECT_OK(ie_network_get_input_shapes(network, &shapes));
    EXPECT_EQ(shapes.shape_num, 1);

    ie_network_input_shapes_free(&shapes);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_reshape, reshape) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    input_shapes_t inputShapes;
    IE_EXPECT_OK(ie_network_get_input_shapes(network, &inputShapes));

    inputShapes.shapes[0].shape.dims[0] = 2;

    IE_EXPECT_OK(ie_network_reshape(network, inputShapes));

    input_shapes_t inputShapes2;
    IE_EXPECT_OK(ie_network_get_input_shapes(network, &inputShapes2));
    EXPECT_EQ(inputShapes2.shapes[0].shape.dims[0], 2);

    ie_network_input_shapes_free(&inputShapes2);
    ie_network_input_shapes_free(&inputShapes);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_outputs_number, getNumber) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    size_t size;
    IE_EXPECT_OK(ie_network_get_outputs_number(network, &size));
    EXPECT_EQ(size, 1);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_output_name, getName) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    char *output_name = nullptr;
    IE_EXPECT_OK(ie_network_get_output_name(network, 0, &output_name));
    EXPECT_STREQ(output_name, "fc_out");

    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_output_name, incorrectNumber) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    char *output_name = nullptr;
    EXPECT_EQ(IEStatusCode::OUT_OF_BOUNDS, ie_network_get_output_name(network, 3, &output_name));

    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_output_precision, getPrecision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "fc_out";
    precision_e p;
    IE_EXPECT_OK(ie_network_get_output_precision(network, name, &p));
    EXPECT_EQ(p, precision_e::FP32);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_set_output_precision, setPrecision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "fc_out";
    precision_e p = precision_e::FP16;
    IE_EXPECT_OK(ie_network_set_output_precision(network, name, p));

    precision_e precision_res;
    IE_EXPECT_OK(ie_network_get_output_precision(network, name, &precision_res));
    EXPECT_EQ(p, precision_res);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_output_layout, getLayout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "fc_out";
    layout_e l;
    IE_EXPECT_OK(ie_network_get_output_layout(network, name, &l));
    EXPECT_EQ(l, layout_e::NC);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_set_output_layout, setLayout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "fc_out";
    layout_e l = layout_e::CN;
    IE_EXPECT_OK(ie_network_set_output_layout(network, name, l));
    layout_e l_res;
    IE_EXPECT_OK(ie_network_get_output_layout(network, name, &l_res));
    EXPECT_EQ(l, l_res);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_network_get_output_dims, getDims) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *name = "fc_out";
    dimensions_t dims_res;
    IE_EXPECT_OK(ie_network_get_output_dims(network, name, &dims_res));
    EXPECT_EQ(dims_res.dims[0], 1);
    EXPECT_EQ(dims_res.dims[1], 10);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_exec_network_create_infer_request, createInferRquest) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network,device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_exec_network_get_config, getConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network,device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_param_t param;
    param.params = nullptr;
    IE_EXPECT_OK(ie_exec_network_get_config(exe_network, "CPU_THREADS_NUM", &param));

    ie_param_free(&param);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_exec_network_set_config, setConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_param_t param;
    if (ie_core_get_metric(core, "GPU", "AVAILABLE_DEVICES", &param) != IEStatusCode::OK) {
        ie_core_free(&core);
        GTEST_SKIP();
    }

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *device_name = "MULTI:GPU,CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_config_t config_param = {"MULTI_DEVICE_PRIORITIES", "GPU,CPU", nullptr};
    IE_EXPECT_OK(ie_exec_network_set_config(exe_network, &config_param));

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
    ie_param_free(&param);
}

TEST(ie_exec_network_get_metric, getMetric) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_param_t param;
    param.params = nullptr;
    IE_EXPECT_OK(ie_exec_network_get_metric(exe_network, "SUPPORTED_CONFIG_KEYS", &param));

    ie_param_free(&param);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_get_blob, getBlob) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    char *input_name = nullptr;
    IE_EXPECT_OK(ie_network_get_input_name(network, 0, &input_name));
    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, input_name, &blob));

    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_name_free(&input_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_set_blob, setBlob) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    dimensions_t dim_t;
    precision_e p = precision_e::U8;
    layout_e l = layout_e::NCHW;
    IE_EXPECT_OK(ie_network_get_input_dims(network, "data", &dim_t));
    IE_EXPECT_OK(ie_network_set_input_layout(network, "data", l));
    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", p));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = p;
    tensor.layout = l;
    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));

    IE_EXPECT_OK(ie_infer_request_set_blob(infer_request, "data", blob));

    ie_blob_deallocate(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_infer, infer) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "data", &blob));


    cv::Mat image = cv::imread(input_image);
    Mat2Blob(image, blob);

    IE_EXPECT_OK(ie_infer_request_infer(infer_request));

    ie_blob_t *output_blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "fc_out", &output_blob));
    dimensions_t dim_res;
    IE_EXPECT_OK(ie_blob_get_dims(output_blob, &dim_res));
    EXPECT_EQ(dim_res.ranks, 2);
    EXPECT_EQ(dim_res.dims[1], 10);

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
    float *output_data = (float *)(buffer.buffer);
    EXPECT_NEAR(output_data[9], 0.f, 1.e-5);

    ie_blob_free(&output_blob);
    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_infer_async, inferAsyncWaitFinish) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "data", &blob));


    cv::Mat image = cv::imread(input_image);
    Mat2Blob(image, blob);

    IE_EXPECT_OK(ie_infer_request_infer_async(infer_request));

    ie_blob_t *output_blob = nullptr;
    if (!HasFatalFailure()) {
        IE_EXPECT_OK(ie_infer_request_wait(infer_request, -1));

        IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "fc_out", &output_blob));
        EXPECT_NE(nullptr, output_blob);

        ie_blob_buffer_t buffer;
        IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
        float *output_data = (float *)(buffer.buffer);
        EXPECT_NEAR(output_data[9], 0.f, 1.e-5);
    }

    ie_blob_free(&output_blob);
    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_infer_async, inferAsyncWaitTime) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "data", &blob));
    EXPECT_NE(nullptr, blob);


    cv::Mat image = cv::imread(input_image);
    Mat2Blob(image, blob);

    IE_EXPECT_OK(ie_infer_request_infer_async(infer_request));

    ie_blob_t *output_blob = nullptr;
    if (!HasFatalFailure()) {
        auto waitStatus = ie_infer_request_wait(infer_request, 10);
        EXPECT_TRUE((IEStatusCode::OK == waitStatus) || (IEStatusCode::RESULT_NOT_READY == waitStatus));
        if (IEStatusCode::RESULT_NOT_READY == waitStatus) {
            IE_EXPECT_OK(ie_infer_request_wait(infer_request, -1));
        }

        IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "fc_out", &output_blob));

        ie_blob_buffer_t buffer;
        IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
        float *output_data = (float *)(buffer.buffer);
        EXPECT_NEAR(output_data[9], 0.f, 1.e-5);
    }

    ie_blob_free(&output_blob);
    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_set_batch, setBatch) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    ie_config_t config = {"DYN_BATCH_ENABLED", "YES", nullptr};
    IE_EXPECT_OK(ie_core_set_config(core, &config, device_name));

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    IE_EXPECT_OK(ie_infer_request_set_batch(infer_request, 1));

    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_set_batch, setZeroBatch) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    ie_config_t config = {"DYN_BATCH_ENABLED", "YES", nullptr};
    IE_EXPECT_OK(ie_core_set_config(core, &config, device_name));

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    EXPECT_EQ(IEStatusCode::GENERAL_ERROR, ie_infer_request_set_batch(infer_request, 0));

    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_infer_request_set_batch, setNegativeBatch) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    ie_config_t config = {"DYN_BATCH_ENABLED", "YES", nullptr};
    IE_EXPECT_OK(ie_core_set_config(core, &config, device_name));

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    EXPECT_EQ(IEStatusCode::GENERAL_ERROR, ie_infer_request_set_batch(infer_request, -1));

    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_blob_make_memory, makeMemory) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_make_memory_from_preallocated, makeMemoryfromPreallocated) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;
    uint8_t array[1][3][4][4]= {0};

    size_t size = 48;
    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor, &array, size, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_free(&blob);
}

TEST(ie_blob_make_memory_with_roi, makeMemorywithROI) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *input_blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &input_blob));
    EXPECT_NE(nullptr, input_blob);

    roi_t roi = {0, 0, 0, 1, 1};
    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_with_roi(input_blob, &roi, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_deallocate(&blob);
    ie_blob_free(&input_blob);
}

TEST(ie_blob_deallocate, blobDeallocate) {
    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_get_dims, getDims) {
    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    dimensions_t dim_res;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dim_res));
    EXPECT_EQ(dim_t.ranks, dim_res.ranks);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_get_layout, getLayout) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    layout_e l;
    IE_EXPECT_OK(ie_blob_get_layout(blob, &l));
    EXPECT_EQ(tensor.layout, l);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_get_precision, getPrecision) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    precision_e p;
    IEStatusCode status3 = ie_blob_get_precision(blob, &p);
    EXPECT_EQ(status3, IEStatusCode::OK);
    EXPECT_EQ(tensor.precision, p);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_size, getSize) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::I16;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    int size_res;
    IE_EXPECT_OK(ie_blob_size(blob, &size_res));
    EXPECT_EQ(size_res, 48);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_byte_size, getByteSize) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::I16;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    int size_res;
    IE_EXPECT_OK(ie_blob_byte_size(blob, &size_res));
    EXPECT_EQ(size_res, 96);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_get_buffer, getBuffer) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_buffer_t blob_buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &blob_buffer));
    EXPECT_NE(nullptr, blob_buffer.buffer);

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_get_cbuffer, getBuffer) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_buffer_t blob_cbuffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &blob_cbuffer));
    EXPECT_NE(nullptr, blob_cbuffer.cbuffer);

    ie_blob_deallocate(&blob);
}

TEST(ie_infer_set_completion_callback, setCallback) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "data", &blob));

    cv::Mat image = cv::imread(input_image);
    Mat2Blob(image, blob);

    ie_complete_call_back_t callback;
    callback.completeCallBackFunc = completion_callback;
    callback.args = infer_request;

    IE_EXPECT_OK(ie_infer_set_completion_callback(infer_request, &callback));

    IE_EXPECT_OK(ie_infer_request_infer_async(infer_request));

    if (!HasFatalFailure()) {
        std::unique_lock<std::mutex> lock(m);
        condVar.wait(lock, []{ return ready; });
    }

    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST(ie_blob_make_memory_nv12, makeNV12Blob) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));
    IE_EXPECT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_free(&blob_nv12);
    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}

TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobFromNullptrBlobs) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));
    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));
}

TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobFromPlanesWithDifferentElementSize) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = precision_e::U8;
    tensor_uv.precision = precision_e::FP32;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}

TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobFromPlanesWithNonU8Precision) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::FP32;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}

TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobWithInconsistentBatchSize) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {2, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}

TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobWithWrongChannelNumber) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_y, &blob_nv12));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_uv, blob_uv, &blob_nv12));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_uv, blob_y, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}

TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobWithWrongHeightRation) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 2, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}


TEST(ie_blob_make_memory_nv12, cannotMakeNV12BlobWithWrongWidthRation) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 4}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));

    IE_EXPECT_NOT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_uv);
}

TEST(ie_blob_make_memory_nv12, NV12BlobInvalidAfterDeallocateYPlane) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));
    IE_EXPECT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_y);
    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob_nv12, &buffer));
    EXPECT_EQ(nullptr, buffer.buffer);

    ie_blob_deallocate(&blob_uv);
    ie_blob_free(&blob_nv12);
}

TEST(ie_blob_make_memory_nv12, NV12BlobInvalidAfterDeallocateUVPlane) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_uv = {4, {1, 2, 4, 6}};
    tensor_desc tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_uv, &blob_uv));
    IE_EXPECT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    ie_blob_deallocate(&blob_uv);
    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob_nv12, &buffer));
    EXPECT_EQ(nullptr, buffer.buffer);

    ie_blob_deallocate(&blob_y);
    ie_blob_free(&blob_nv12);
}

TEST(ie_blob_make_memory_nv12, inferRequestWithNV12Blob) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));
    IE_EXPECT_OK(ie_network_set_input_layout(network, "data", layout_e::NCHW));
    IE_EXPECT_OK(ie_network_set_input_resize_algorithm(network, "data", resize_alg_e::RESIZE_BILINEAR));
    IE_EXPECT_OK(ie_network_set_color_format(network, "data", colorformat_e::NV12));
    IE_EXPECT_OK(ie_network_set_output_precision(network, "fc_out", precision_e::FP32));

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    size_t img_width = 224, img_height = 224;
    size_t img_size = img_width * (img_height * 3 / 2);
    unsigned char *img_data = (unsigned char *)calloc(img_size, sizeof(unsigned char));
    EXPECT_NE(nullptr, img_data);
    EXPECT_EQ(img_size, read_image_from_file(input_image_nv12, img_data, img_size));

    dimensions_t dim_y = {4, {1, 1, img_height, img_width}};
    dimensions_t dim_uv = {4, {1, 2, img_height / 2, img_width / 2}};
    tensor_desc_t tensor_y, tensor_uv;
    tensor_y.dims = dim_y;
    tensor_uv.dims = dim_uv;
    tensor_y.precision = tensor_uv.precision = precision_e::U8;
    tensor_y.layout = tensor_uv.layout = layout_e::NHWC;
    const size_t offset = img_width * img_height;

    ie_blob_t *blob_y = nullptr, *blob_uv = nullptr, *blob_nv12 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor_y, img_data, img_width * img_height, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor_uv, img_data + offset, img_width * (img_height / 2), &blob_uv));
    IE_EXPECT_OK(ie_blob_make_memory_nv12(blob_y, blob_uv, &blob_nv12));

    IE_EXPECT_OK(ie_infer_request_set_blob(infer_request, "data", blob_nv12));
    IE_EXPECT_OK(ie_infer_request_infer(infer_request));

    ie_blob_t *output_blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "fc_out", &output_blob));
    EXPECT_NE(nullptr, output_blob);

    ie_blob_buffer_t buffer;
    buffer.buffer = nullptr;
    IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
    EXPECT_NE(buffer.buffer, nullptr);
    if (buffer.buffer) {
        float *output_data = (float *)(buffer.buffer);
        EXPECT_NEAR(output_data[1], 0.f, 1.e-5);
    }

    ie_blob_free(&output_blob);
    ie_blob_free(&blob_nv12);
    ie_blob_free(&blob_uv);
    ie_blob_free(&blob_y);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
    free(img_data);
}

TEST(ie_blob_make_memory_i420, makeI420Blob) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_free(&blob_i420);
    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromNullptrBlobs) {
    dimensions_t dim = {4, {1, 1, 8, 12}};
    tensor_desc tensor;
    tensor.dims = dim;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NHWC;

    ie_blob_t *blob = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor, &blob));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob, nullptr, nullptr, &blob_i420));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(nullptr, blob, nullptr, &blob_i420));

    ie_blob_deallocate(&blob);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromPlanesWithDifferentElementSize) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = precision_e::U8;
    tensor_u.precision = tensor_v.precision = precision_e::FP32;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromPlanesWithNonU8Precision) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::FP32;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromPlanesWithInconsistentBatchSize) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {2, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromPlanesWithWrongChannelNumber) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 2, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromPlanesWithWrongWidthRatio) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 4}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, cannotMakeI420BlobFromPlanesWithWrongHeightRatio) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 2, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_NOT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
}

TEST(ie_blob_make_memory_i420, I420BlobInvalidAfterDeallocateYPlane) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_y);
    ie_blob_buffer_t i420_buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob_i420, &i420_buffer));
    EXPECT_EQ(nullptr, i420_buffer.buffer);

    ie_blob_deallocate(&blob_u);
    ie_blob_deallocate(&blob_v);
    ie_blob_free(&blob_i420);
}

TEST(ie_blob_make_memory_i420, I420BlobInvalidAfterDeallocateUPlane) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_u);
    ie_blob_buffer_t i420_buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob_i420, &i420_buffer));
    EXPECT_EQ(nullptr, i420_buffer.buffer);

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_v);
    ie_blob_free(&blob_i420);
}

TEST(ie_blob_make_memory_i420, I420BlobInvalidAfterDeallocateVPlane) {
    dimensions_t dim_y = {4, {1, 1, 8, 12}}, dim_u = {4, {1, 1, 4, 6}}, dim_v = {4, {1, 1, 4, 6}};
    tensor_desc tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_y, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_u, &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory(&tensor_v, &blob_v));
    IE_EXPECT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    ie_blob_deallocate(&blob_v);
    ie_blob_buffer_t i420_buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob_i420, &i420_buffer));
    EXPECT_EQ(nullptr, i420_buffer.buffer);

    ie_blob_deallocate(&blob_y);
    ie_blob_deallocate(&blob_u);
    ie_blob_free(&blob_i420);
}

TEST(ie_blob_make_memory_i420, inferRequestWithI420) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml, bin, &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, "data", precision_e::U8));
    IE_EXPECT_OK(ie_network_set_input_layout(network, "data", layout_e::NCHW));
    IE_EXPECT_OK(ie_network_set_input_resize_algorithm(network, "data", resize_alg_e::RESIZE_BILINEAR));
    IE_EXPECT_OK(ie_network_set_color_format(network, "data", colorformat_e::I420));
    IE_EXPECT_OK(ie_network_set_output_precision(network, "fc_out", precision_e::FP32));

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    size_t img_width = 224, img_height = 224;
    size_t img_size = img_width * (img_height * 3 / 2);
    unsigned char *img_data = (unsigned char *)calloc(img_size, sizeof(unsigned char));
    EXPECT_NE(nullptr, img_data);
    EXPECT_EQ(img_size, read_image_from_file(input_image_nv12, img_data, img_size));

    dimensions_t dim_y = {4, {1, 1, img_height, img_width}};
    dimensions_t dim_u = {4, {1, 1, img_height / 2, img_width / 2}};
    dimensions_t dim_v = {4, {1, 1, img_height / 2, img_width / 2}};
    tensor_desc_t tensor_y, tensor_u, tensor_v;
    tensor_y.dims = dim_y;
    tensor_u.dims = dim_u;
    tensor_v.dims = dim_v;
    tensor_y.precision = tensor_u.precision = tensor_v.precision = precision_e::U8;
    tensor_y.layout = tensor_u.layout = tensor_v.layout = layout_e::NHWC;
    const size_t offset = img_width * img_height;

    ie_blob_t *blob_y = nullptr, *blob_u = nullptr, *blob_v = nullptr, *blob_i420 = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor_y, img_data, img_width * img_height, &blob_y));
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor_u, img_data + offset, img_width * (img_height / 4), &blob_u));
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor_v, img_data + offset * 5 / 4, img_width * (img_height / 4), &blob_v));
    IE_EXPECT_OK(ie_blob_make_memory_i420(blob_y, blob_u, blob_v, &blob_i420));

    IE_EXPECT_OK(ie_infer_request_set_blob(infer_request, "data", blob_i420));
    IE_EXPECT_OK(ie_infer_request_infer(infer_request));

    ie_blob_t *output_blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "fc_out", &output_blob));

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
    float *output_data = (float *)(buffer.buffer);
    EXPECT_NEAR(output_data[1], 0.f, 1.e-5);

    ie_blob_free(&output_blob);
    ie_blob_free(&blob_i420);
    ie_blob_free(&blob_v);
    ie_blob_free(&blob_u);
    ie_blob_free(&blob_y);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
    free(img_data);
}

int main(int argc, char *argv[]){
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
