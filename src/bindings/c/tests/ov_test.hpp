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

#ifndef ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined __INTEL_COMPILER || defined _MSC_VER
#            define ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#        define ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#    define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <wchar.h>
#endif

#ifdef _WIN32
#    include <windows.h>
#else
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        include <codecvt>
#        include <locale>
#    endif
#endif

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
extern const std::vector<std::wstring> test_unicode_postfix_vector;
inline std::string wstring_to_string(const std::wstring& wstr) {
#    ifdef _WIN32
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#    endif
}

inline std::wstring string_to_wstring(const std::string& string) {
    const char* str = string.c_str();
#    ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#    endif
}

inline void fix_slashes(std::string& str) {
    std::replace(str.begin(), str.end(), '/', '\\');
}

inline void fix_slashes(std::wstring& str) {
    std::replace(str.begin(), str.end(), L'/', L'\\');
}

inline bool copy_file(std::string source_path, std::wstring dest_path) {
#    ifndef _WIN32
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream dest(wstring_to_string(dest_path), std::ios::binary);
#    else
    fix_slashes(source_path);
    fix_slashes(dest_path);
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream dest(dest_path, std::ios::binary);
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
    auto result = string_to_wstring(source_path);
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
        result = remove(wstring_to_string(path).c_str());
#    endif
    }
    (void)result;
}
#endif
