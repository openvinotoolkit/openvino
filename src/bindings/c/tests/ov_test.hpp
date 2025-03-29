// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <condition_variable>
#include <fstream>
#include <openvino/util/file_util.hpp>

#include "openvino/c/openvino.h"
#include "openvino/openvino.hpp"
#include "test_model_repo.hpp"

#define OV_EXPECT_OK(...)           EXPECT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_ASSERT_OK(...)           ASSERT_EQ(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_NOT_OK(...)       EXPECT_NE(ov_status_e::OK, __VA_ARGS__)
#define OV_EXPECT_ARREQ(arr1, arr2) EXPECT_TRUE(std::equal(std::begin(arr1), std::end(arr1), std::begin(arr2)))

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <wchar.h>
#endif

extern std::map<ov_element_type_e, size_t> element_type_size_map;
#define GET_ELEMENT_TYPE_SIZE(a) element_type_size_map[a]

class ov_capi_test_base : public ::testing::TestWithParam<std::string> {
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
    std::string xml_file_name, bin_file_name;
};

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
            size_t file_len = is.tellg();
            result.resize(file_len + 1);
            if (file_len > 0) {
                is.seekg(0, std::ifstream::beg);
                is.read(reinterpret_cast<char*>(&result[0]), file_len);
            }
        }
    }
    return result;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
inline void fix_slashes(std::string& str) {
    std::replace(str.begin(), str.end(), '/', '\\');
}

inline void fix_slashes(std::wstring& str) {
    std::replace(str.begin(), str.end(), L'/', L'\\');
}

inline bool copy_file(std::string source_path, std::wstring dest_path) {
#    ifdef _WIN32
    fix_slashes(source_path);
    fix_slashes(dest_path);
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream dest(dest_path, std::ios::binary);
#    else
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream dest(ov::util::wstring_to_string(dest_path), std::ios::binary);
#    endif
    bool result = source && dest;
    std::istreambuf_iterator<char> begin_source(source);
    std::istreambuf_iterator<char> end_source;
    std::ostreambuf_iterator<char> begin_dest(dest);
    copy(begin_source, end_source, begin_dest);

    source.close();
    dest.close();
    return result;
}

inline std::wstring add_unicode_postfix_to_path(std::string source_path, std::wstring postfix) {
    fix_slashes(source_path);
    auto result = ov::util::string_to_wstring(source_path);
    auto extPos = result.rfind('.');
    auto extension = result.substr(extPos, result.size());
    auto file_name = result.substr(0, extPos);

    return file_name + postfix + extension;
}

inline void remove_file_ws(std::wstring path) {
    int result = 0;
    if (!path.empty()) {
#    ifdef _WIN32
        result = _wremove(path.c_str());
#    else
        result = remove(ov::util::wstring_to_string(path).c_str());
#    endif
    }
    (void)result;
}
#endif
