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

namespace TestDataHelpers {

    static const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
            '\\';
#else
            '/';
#endif

    std::string getModelPathNonFatal() noexcept {
        if (const auto envVar = std::getenv("MODELS_PATH")) {
            return envVar;
        }

#ifdef MODELS_PATH
        return MODELS_PATH;
#else
        return "";
#endif
    }

    std::string get_models_path() {
        return getModelPathNonFatal() + kPathSeparator + std::string("models");
    };

    std::string get_data_path() {
        if (const auto envVar = std::getenv("DATA_PATH")) {
            return envVar;
        }

#ifdef DATA_PATH
        return DATA_PATH;
#else
        return "";
#endif
    }

    std::string generate_model_path(std::string dir, std::string filename) {
        return get_models_path() + kPathSeparator + dir + kPathSeparator + filename;
    }

    std::string generate_image_path(std::string dir, std::string filename) {
        return get_data_path() + kPathSeparator + "validation_set" + kPathSeparator + dir + kPathSeparator + filename;
    }

    std::string generate_ieclass_xml_path(std::string filename) {
        return getModelPathNonFatal() + kPathSeparator + "ie_class" + kPathSeparator + filename;
    }
} // namespace TestDataHelpers


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
#    ifdef __MINGW32__
std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins_mingw.xml");
#    else
std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins_win.xml");
#    endif
#elif defined __APPLE__
std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins_apple.xml");
#else
std::string plugins_xml_std = TestDataHelpers::generate_ieclass_xml_path("plugins.xml");
#endif
const char* plugins_xml = plugins_xml_std.c_str();

#define OV_EXPECT_OK(...)           EXPECT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_ASSERT_OK(...)           ASSERT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_NOT_OK(...)       EXPECT_NE(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_ARREQ(arr1, arr2) EXPECT_TRUE(std::equal(std::begin(arr1), std::end(arr1), std::begin(arr2)))

std::map<ov_element_type_e, size_t> element_type_size_map = {{ov_element_type_e::BOOLEAN, 8},
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

size_t find_device(ov_available_devices_t avai_devices, const char* device_name) {
    for (size_t i = 0; i < avai_devices.num_devices; ++i) {
        if (strstr(avai_devices.devices[i], device_name))
            return i;
    }

    return -1;
}

static std::vector<uint8_t> content_from_file(const char* filename, bool is_binary) {
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
