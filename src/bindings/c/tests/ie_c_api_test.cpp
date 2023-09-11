// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <condition_variable>
#include <fstream>
#include "inference_engine.hpp"
#include <mutex>

#include "test_model_repo.hpp"

#define IE_EXPECT_OK(...) EXPECT_EQ(IEStatusCode::OK, __VA_ARGS__)
#define IE_ASSERT_OK(...) ASSERT_EQ(IEStatusCode::OK, __VA_ARGS__)
#define IE_EXPECT_NOT_OK(...) EXPECT_NE(IEStatusCode::OK, __VA_ARGS__)

OPENVINO_SUPPRESS_DEPRECATED_START

#include <c_api/ie_c_api.h>

static std::mutex m;
static bool ready = false;
static std::condition_variable condVar;

static void completion_callback(void* args) {
    ie_infer_request_t* infer_request = (ie_infer_request_t*)args;
    ie_blob_t* output_blob = nullptr;

    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, "Relu_1", &output_blob));

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &buffer));
    float* output_data = (float*)(buffer.buffer);
    EXPECT_NEAR(output_data[9], 0.f, 1.e-5);

    ie_blob_free(&output_blob);

    std::lock_guard<std::mutex> lock(m);
    ready = true;
    condVar.notify_one();
}

class ie_c_api_test : public ::testing::TestWithParam<std::string> {
public:
    void SetUp() override {
        TestDataHelpers::generate_test_model();
        xml_file_name = TestDataHelpers::get_model_xml_file_name();
        bin_file_name = TestDataHelpers::get_model_bin_file_name();
    }

    void TearDown() override {
        TestDataHelpers::release_test_model();
    }

public:
    size_t find_device(ie_available_devices_t avai_devices, const char* device_name) {
        for (size_t i = 0; i < avai_devices.num_devices; ++i) {
            if (strstr(avai_devices.devices[i], device_name))
                return i;
        }

        return -1;
    }

    std::vector<uint8_t> content_from_file(const char* filename, bool is_binary) {
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

    std::string xml_file_name, bin_file_name;
    const char* input_port_name = "Param_1";
    const char* output_port_name = "Relu_1";
};

INSTANTIATE_TEST_SUITE_P(ie_c_api, ie_c_api_test, ::testing::Values(""));

TEST_P(ie_c_api_test, ie_c_api_version) {
    ie_version_t version = ie_c_api_version();
    auto ver = InferenceEngine::GetInferenceEngineVersion();
    std::string ver_str = ver->buildNumber;

    EXPECT_EQ(strcmp(version.api_version, ver_str.c_str()), 0);
    ie_version_free(&version);
}

TEST_P(ie_c_api_test, ie_core_create_coreCreatewithConfig) {
    std::string plugins_xml = TestDataHelpers::generate_test_xml_file();
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create(plugins_xml.c_str(), &core));
    ASSERT_NE(nullptr, core);

    ie_core_free(&core);
    TestDataHelpers::delete_test_xml_file();
}

TEST_P(ie_c_api_test, ie_core_create_coreCreateNoConfig) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_get_available_devices) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));

    ie_available_devices_t avai_devices = {0};
    IE_EXPECT_OK(ie_core_get_available_devices(core, &avai_devices));

    ie_core_available_devices_free(&avai_devices);
    ie_core_free(&core);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

TEST_P(ie_c_api_test, ie_core_register_plugin) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *plugin_name = "test_plugin";
    const char *device_name = "BLA";
    IE_EXPECT_OK(ie_core_register_plugin(core, plugin_name, device_name));

    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_register_plugins) {
    std::string plugins_xml = TestDataHelpers::generate_test_xml_file();
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    IE_EXPECT_OK(ie_core_register_plugins(core, plugins_xml.c_str()));

    // Trigger plugin loading
    ie_core_versions_t versions = {0};
    IE_EXPECT_OK(ie_core_get_versions(core, "CUSTOM", &versions));
    ie_core_versions_free(&versions);

    ie_core_free(&core);
    TestDataHelpers::delete_test_xml_file();
}

TEST_P(ie_c_api_test, ie_core_unload_plugin) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    ie_core_versions_t versions = {0};
    // Trigger plugin loading
    IE_EXPECT_OK(ie_core_get_versions(core, device_name, &versions));
    // Unload plugin
    IE_EXPECT_OK(ie_core_unregister_plugin(core, device_name));

    ie_core_versions_free(&versions);
    ie_core_free(&core);
}

#endif // !OPENVINO_STATIC_LIBRARY

TEST_P(ie_c_api_test, ie_core_set_config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    const char *device_name = "CPU";
    ie_config_t config = {"CPU_THREADS_NUM", "3", nullptr};
    IE_EXPECT_OK(ie_core_set_config(core, &config, device_name));

    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_get_metric) {
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

TEST_P(ie_c_api_test, ie_core_get_config) {
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

TEST_P(ie_c_api_test, ie_core_get_versions) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_core_versions_t versions = {0};
    IE_EXPECT_OK(ie_core_get_versions(core, "CPU", &versions));
    EXPECT_EQ(versions.num_vers, 1);

    ie_core_versions_free(&versions);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_read_network) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_read_network_from_memory) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    std::vector<uint8_t> weights_content(content_from_file(bin_file_name.c_str(), true));

    tensor_desc_t weights_desc { ANY, { 1, { weights_content.size() } }, U8 };
    ie_blob_t *weights_blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&weights_desc, weights_content.data(), weights_content.size(), &weights_blob));
    EXPECT_NE(nullptr, weights_blob);

    if (weights_blob != nullptr) {
        std::vector<uint8_t> xml_content(content_from_file(xml_file_name.c_str(), false));

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

TEST_P(ie_c_api_test, ie_core_export_network_to_file) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;

    IE_EXPECT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "HETERO:CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    std::string export_path = TestDataHelpers::get_exported_blob_file_name();
    IE_EXPECT_OK(ie_core_export_network(exe_network, export_path.c_str()));
    std::ifstream file(export_path.c_str());
    EXPECT_NE(file.peek(), std::ifstream::traits_type::eof());

    EXPECT_NE(nullptr, exe_network);
    ie_exec_network_free(&exe_network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_import_network_from_memory) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_executable_network_t *exe_network = nullptr;

    IE_EXPECT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "HETERO:CPU", nullptr, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    std::string export_path = TestDataHelpers::get_exported_blob_file_name();
    IE_EXPECT_OK(ie_core_export_network(exe_network, export_path.c_str()));

    std::vector<uint8_t> buffer(content_from_file(export_path.c_str(), true));
    ie_executable_network_t *network = nullptr;

    IE_EXPECT_OK(ie_core_import_network_from_memory(core, buffer.data(), buffer.size(), "HETERO:CPU", nullptr, &network));
    EXPECT_NE(nullptr, network);

    ie_exec_network_free(&network);
    ie_exec_network_free(&exe_network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_import_network_from_file) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_config_t conf = {nullptr, nullptr, nullptr};

    ie_executable_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "HETERO:CPU", &conf, &network));
    EXPECT_NE(nullptr, network);

    std::string exported_model = TestDataHelpers::get_exported_blob_file_name();
    IE_EXPECT_OK(ie_core_export_network(network, exported_model.c_str()));
    std::ifstream file(exported_model);
    EXPECT_NE(file.peek(), std::ifstream::traits_type::eof());

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_import_network(core, exported_model.c_str(), "HETERO:CPU", &conf, &exe_network));
    EXPECT_NE(nullptr, exe_network);

     ie_exec_network_free(&network);
    ie_exec_network_free(&exe_network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_import_network_from_file_errorHandling) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_config_t config = {nullptr, nullptr, nullptr};

    ie_executable_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "HETERO:CPU", &config, &network));
    EXPECT_NE(nullptr, network);

    std::string exported_model = TestDataHelpers::get_exported_blob_file_name();
    IE_EXPECT_OK(ie_core_export_network(network, exported_model.c_str()));

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_NOT_OK(ie_core_import_network(core, nullptr, "HETERO:CPU", &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_import_network(core, exported_model.c_str(), nullptr, &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_import_network(core, exported_model.c_str(), "HETERO:CPU", &config, nullptr));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_import_network(core, exported_model.c_str(), "UnregisteredDevice", &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_OK(ie_core_import_network(core, exported_model.c_str(), "HETERO:CPU", nullptr, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&network);
    ie_exec_network_free(&exe_network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_load_network_with_config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_layout(network, input_port_name, layout_e::NHWC));
    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, precision_e::U8));

    ie_config_t config = {"CPU_THREADS_NUM", "3", nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_load_network_no_config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_load_network_null_Config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, "CPU", nullptr, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_load_network_from_file_no_config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "CPU", &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_core_load_network_from_file_null_config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "CPU", nullptr, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_exec_network_free(&exe_network);
    ie_core_free(&core);
}


TEST_P(ie_c_api_test, ie_core_load_network_from_file_errorHandling) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_NOT_OK(ie_core_load_network_from_file(nullptr, xml_file_name.c_str(), "CPU", &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_load_network_from_file(core, nullptr, "CPU", &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), nullptr, &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "CPU", &config, nullptr));
    EXPECT_EQ(nullptr, exe_network);

    IE_EXPECT_NOT_OK(ie_core_load_network_from_file(core, xml_file_name.c_str(), "UnregisteredDevice", &config, &exe_network));
    EXPECT_EQ(nullptr, exe_network);

    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_name) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    char *network_name = nullptr;
    IE_EXPECT_OK(ie_network_get_name(network, &network_name));

    EXPECT_NE(network_name, nullptr);

    ie_network_name_free(&network_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_inputs_number) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    size_t size;
    IEStatusCode status_result = ie_network_get_inputs_number(network, &size);
    EXPECT_EQ(status_result, IEStatusCode::OK);
    EXPECT_EQ(size, 1);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_name) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    char *input_name = nullptr;
    IE_EXPECT_OK(ie_network_get_input_name(network, 0, &input_name));

    EXPECT_STREQ(input_name, input_port_name);

    ie_network_name_free(&input_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_precision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    precision_e p;
    IE_EXPECT_OK(ie_network_get_input_precision(network, input_port_name, &p));
    EXPECT_EQ(p, precision_e::FP32);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_precision_incorrectName) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    const char *name = "model";
    precision_e p;
    EXPECT_EQ(IEStatusCode::NOT_FOUND, ie_network_get_input_precision(network, name, &p));

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_set_input_precision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    const precision_e p = precision_e::FP16;
    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, p));
    precision_e p2;
    IE_EXPECT_OK(ie_network_get_input_precision(network, input_port_name, &p2));
    EXPECT_EQ(p, p2);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_layout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    layout_e l;
    IE_EXPECT_OK(ie_network_get_input_layout(network, input_port_name, &l));
    EXPECT_EQ(l, layout_e::NCHW);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_set_input_layout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    const layout_e l = layout_e ::NHWC;
    IE_EXPECT_OK(ie_network_set_input_layout(network, input_port_name, l));
    layout_e l2;
    IE_EXPECT_OK(ie_network_get_input_layout(network, input_port_name, &l2));
    EXPECT_EQ(l, l2);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_dims) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    dimensions_t dims_res;
    IE_EXPECT_OK(ie_network_get_input_dims(network, input_port_name, &dims_res));
    EXPECT_EQ(dims_res.dims[0], 1);
    EXPECT_EQ(dims_res.dims[1], 3);
    EXPECT_EQ(dims_res.dims[2], 227);
    EXPECT_EQ(dims_res.dims[3], 227);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_resize_algorithm_resize_algo) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    resize_alg_e resizeAlg;
    IE_EXPECT_OK(ie_network_get_input_resize_algorithm(network, input_port_name, &resizeAlg));
    EXPECT_EQ(resizeAlg, resize_alg_e::NO_RESIZE);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_set_input_resize_algorithm_resize_algo) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    resize_alg_e resizeAlg = resize_alg_e::RESIZE_BILINEAR;
    IE_EXPECT_OK(ie_network_set_input_resize_algorithm(network, input_port_name, resizeAlg));

    resize_alg_e resizeAlg2;
    IE_EXPECT_OK(ie_network_get_input_resize_algorithm(network, input_port_name, &resizeAlg2));
    EXPECT_EQ(resizeAlg, resizeAlg2);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_color_format) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    colorformat_e color;
    IE_EXPECT_OK(ie_network_get_color_format(network, input_port_name, &color));
    EXPECT_EQ(color, colorformat_e::RAW);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_set_color_format) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    const colorformat_e color = colorformat_e::BGR;
    IE_EXPECT_OK(ie_network_set_color_format(network, input_port_name, color));

    colorformat_e color2;
    IE_EXPECT_OK(ie_network_get_color_format(network, input_port_name, &color2));
    EXPECT_EQ(color2, colorformat_e::BGR);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_input_shapes) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    input_shapes_t shapes;
    IE_EXPECT_OK(ie_network_get_input_shapes(network, &shapes));
    EXPECT_EQ(shapes.shape_num, 1);

    ie_network_input_shapes_free(&shapes);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_reshape) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
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

TEST_P(ie_c_api_test, ie_network_get_outputs_number) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    size_t size;
    IE_EXPECT_OK(ie_network_get_outputs_number(network, &size));
    EXPECT_EQ(size, 1);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_output_name) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    char *output_name = nullptr;
    IE_EXPECT_OK(ie_network_get_output_name(network, 0, &output_name));
    EXPECT_STREQ(output_name, output_port_name);

    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_output_name_incorrectNumber) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    char *output_name = nullptr;
    EXPECT_EQ(IEStatusCode::OUT_OF_BOUNDS, ie_network_get_output_name(network, 3, &output_name));

    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_output_precision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    precision_e p;
    IE_EXPECT_OK(ie_network_get_output_precision(network, output_port_name, &p));
    EXPECT_EQ(p, precision_e::FP32);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_set_output_precision) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    precision_e p = precision_e::FP16;
    IE_EXPECT_OK(ie_network_set_output_precision(network, output_port_name, p));

    precision_e precision_res;
    IE_EXPECT_OK(ie_network_get_output_precision(network, output_port_name, &precision_res));
    EXPECT_EQ(p, precision_res);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_output_layout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    layout_e l;
    IE_EXPECT_OK(ie_network_get_output_layout(network, output_port_name, &l));
    EXPECT_EQ(l, layout_e::NCHW);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_set_output_layout) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    layout_e l = layout_e::NCHW;
    IE_EXPECT_OK(ie_network_set_output_layout(network, output_port_name, l));
    layout_e l_res;
    IE_EXPECT_OK(ie_network_get_output_layout(network, output_port_name, &l_res));
    EXPECT_EQ(l, l_res);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_network_get_output_dims) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    dimensions_t dims_res;
    IE_EXPECT_OK(ie_network_get_output_dims(network, output_port_name, &dims_res));
    EXPECT_EQ(dims_res.dims[0], 1);
    EXPECT_EQ(dims_res.dims[1], 4);

    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_exec_network_create_infer_request) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
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

TEST_P(ie_c_api_test, ie_exec_network_get_config) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
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

TEST_P(ie_c_api_test, ie_exec_network_get_metric) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
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

TEST_P(ie_c_api_test, ie_infer_request_get_blob) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
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

TEST_P(ie_c_api_test, ie_infer_request_set_blob) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    dimensions_t dim_t;
    precision_e p = precision_e::U8;
    layout_e l = layout_e::NCHW;
    IE_EXPECT_OK(ie_network_get_input_dims(network, input_port_name, &dim_t));
    IE_EXPECT_OK(ie_network_set_input_layout(network, input_port_name, l));
    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, p));

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

    IE_EXPECT_OK(ie_infer_request_set_blob(infer_request, input_port_name, blob));

    ie_blob_deallocate(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_infer_request_infer) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, input_port_name, &blob));

    dimensions_t dims;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dims));
    const size_t blob_elems_count = dims.dims[0] * dims.dims[1] * dims.dims[2] * dims.dims[3];

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &buffer));
    auto* blob_internal_buffer = (uint8_t*)buffer.buffer;
    std::fill(blob_internal_buffer, blob_internal_buffer + blob_elems_count, uint8_t{0});

    IE_EXPECT_OK(ie_infer_request_infer(infer_request));

    ie_blob_t *output_blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, output_port_name, &output_blob));
    dimensions_t dim_res;
    IE_EXPECT_OK(ie_blob_get_dims(output_blob, &dim_res));
    EXPECT_EQ(dim_res.ranks, 4);
    EXPECT_EQ(dim_res.dims[1], 4);

    ie_blob_buffer_t out_buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &out_buffer));
    float *output_data = (float *)(out_buffer.buffer);
    EXPECT_NEAR(output_data[9], 0.f, 1.e-5);

    ie_blob_free(&output_blob);
    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_infer_request_infer_async_wait_finish) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, input_port_name, &blob));

    dimensions_t dims;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dims));
    const size_t blob_elems_count = dims.dims[0] * dims.dims[1] * dims.dims[2] * dims.dims[3];

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &buffer));
    auto* blob_internal_buffer = (uint8_t*)buffer.buffer;
    std::fill(blob_internal_buffer, blob_internal_buffer + blob_elems_count, uint8_t{0});

    IE_EXPECT_OK(ie_infer_request_infer_async(infer_request));

    ie_blob_t *output_blob = nullptr;
    if (!HasFatalFailure()) {
        IE_EXPECT_OK(ie_infer_request_wait(infer_request, -1));

        IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, output_port_name, &output_blob));
        EXPECT_NE(nullptr, output_blob);

        ie_blob_buffer_t out_buffer;
        IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &out_buffer));
        float *output_data = (float *)(out_buffer.buffer);
        EXPECT_NEAR(output_data[9], 0.f, 1.e-5);
    }

    ie_blob_free(&output_blob);
    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_infer_request_infer_async_wait_time) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, input_port_name, &blob));
    EXPECT_NE(nullptr, blob);

    dimensions_t dims;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dims));
    const size_t blob_elems_count = dims.dims[0] * dims.dims[1] * dims.dims[2] * dims.dims[3];

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &buffer));
    auto* blob_internal_buffer = (uint8_t*)buffer.buffer;
    std::fill(blob_internal_buffer, blob_internal_buffer + blob_elems_count, uint8_t{0});

    IE_EXPECT_OK(ie_infer_request_infer_async(infer_request));

    ie_blob_t *output_blob = nullptr;
    if (!HasFatalFailure()) {
        auto waitStatus = ie_infer_request_wait(infer_request, 10);
        EXPECT_TRUE((IEStatusCode::OK == waitStatus) || (IEStatusCode::RESULT_NOT_READY == waitStatus));
        if (IEStatusCode::RESULT_NOT_READY == waitStatus) {
            IE_EXPECT_OK(ie_infer_request_wait(infer_request, -1));
        }

        IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, output_port_name, &output_blob));

        ie_blob_buffer_t out_buffer;
        IE_EXPECT_OK(ie_blob_get_buffer(output_blob, &out_buffer));
        float *output_data = (float *)(out_buffer.buffer);
        EXPECT_NEAR(output_data[9], 0.f, 1.e-5);
    }

    ie_blob_free(&output_blob);
    ie_blob_free(&blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
}

TEST_P(ie_c_api_test, ie_blob_make_memory) {

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

TEST_P(ie_c_api_test, ie_blob_make_memory_from_preallocated) {

    dimensions_t dim_t;
    dim_t.ranks = 4 ;
    dim_t.dims[0] = 1, dim_t.dims[1] = 3, dim_t.dims[2] = 4, dim_t.dims[3] = 4;
    tensor_desc tensor;
    tensor.dims = dim_t ;
    tensor.precision = precision_e::U8;
    tensor.layout = layout_e::NCHW;
    uint8_t array[1][3][4][4]= {{{{0}}}};

    size_t size = 48;
    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_blob_make_memory_from_preallocated(&tensor, &array, size, &blob));
    EXPECT_NE(nullptr, blob);

    ie_blob_free(&blob);
}

TEST_P(ie_c_api_test, ie_blob_make_memory_with_roi) {

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

TEST_P(ie_c_api_test, ie_blob_deallocate) {
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

TEST_P(ie_c_api_test, ie_blob_get_dims) {
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

TEST_P(ie_c_api_test, ie_blob_get_layout) {

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

TEST_P(ie_c_api_test, ie_blob_get_precision) {

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

TEST_P(ie_c_api_test, ie_blob_size) {

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

TEST_P(ie_c_api_test, ie_blob_byte_size) {

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

TEST_P(ie_c_api_test, ie_blob_get_buffer) {

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

TEST_P(ie_c_api_test, ie_blob_get_cbuffer) {

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

TEST_P(ie_c_api_test, ie_infer_set_completion_callback) {
    ie_core_t *core = nullptr;
    IE_ASSERT_OK(ie_core_create("", &core));
    ASSERT_NE(nullptr, core);

    ie_network_t *network = nullptr;
    IE_EXPECT_OK(ie_core_read_network(core, xml_file_name.c_str(), bin_file_name.c_str(), &network));
    EXPECT_NE(nullptr, network);

    IE_EXPECT_OK(ie_network_set_input_precision(network, input_port_name, precision_e::U8));

    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    IE_EXPECT_OK(ie_core_load_network(core, network, device_name, &config, &exe_network));
    EXPECT_NE(nullptr, exe_network);

    ie_infer_request_t *infer_request = nullptr;
    IE_EXPECT_OK(ie_exec_network_create_infer_request(exe_network, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    ie_blob_t *blob = nullptr;
    IE_EXPECT_OK(ie_infer_request_get_blob(infer_request, input_port_name, &blob));

    dimensions_t dims;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dims));
    const size_t blob_elems_count = dims.dims[0] * dims.dims[1] * dims.dims[2] * dims.dims[3];

    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &buffer));
    auto* blob_internal_buffer = (uint8_t*)buffer.buffer;
    std::fill(blob_internal_buffer, blob_internal_buffer + blob_elems_count, uint8_t{0});

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

OPENVINO_SUPPRESS_DEPRECATED_END