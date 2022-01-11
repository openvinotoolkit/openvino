// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// avoiding clash of the "max" macro with std::max
#define NOMINMAX

#include <algorithm>
#include <cmath>
#include <cctype>
#include <math.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <legacy/ie_layers.h>
#include <ie_blob.h>
#include <ie_input_info.hpp>

#include "test_model_repo.hpp"
#include "test_model_path.hpp"
#include <tests_file_utils.hpp>
#include <chrono>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "common_test_utils/test_constants.hpp"

template <class T>
inline std::string to_string_c_locale(T value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

class TestsCommon : public ::testing::Test {
protected:
    void SetUp() override;

    static std::string make_so_name(const std::string & input) {
        return CommonTestUtils::pre + input + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

    void TearDown() override;

public:
    inline std::string get_mock_engine_name() {
        return make_plugin_name("mock_engine");
    }

    static std::string make_plugin_name(const std::string & input) {
        return make_so_name(input);
    }

    static void fill_data(InferenceEngine::Blob::Ptr& blob) {
        fill_data(blob->buffer().as<float*>(), blob->byteSize() / sizeof(float));
    }

    static void fill_data(float *data, size_t size, size_t duty_ratio = 10) {
        for (size_t i = 0; i < size; i++) {
            if ( ( i / duty_ratio)%2 == 1) {
                data[i] = 0.0;
            } else {
                data[i] = sin((float)i);
            }
        }
    }

    static void fill_data_dbgval(float *data, size_t size, float alpha = 1.0f) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i * alpha;
        }
    }

    static void compare(
            InferenceEngine::Blob &res,
            InferenceEngine::Blob &ref,
            float max_diff = 0.01,
            const std::string assertDetails = "") {
        float *res_ptr = res.buffer().as<float*>();
        size_t res_size = res.size();

        float *ref_ptr = ref.buffer().as<float*>();
        size_t ref_size = ref.size();

        ASSERT_EQ(res_size, ref_size) << assertDetails;

        for (size_t i = 0; i < ref_size; i++) {
            ASSERT_NEAR(res_ptr[i], ref_ptr[i], max_diff) << assertDetails;
        }
    }

    static void compare(
            const float* res,
            const float* ref,
            size_t size,
            float max_diff = 0.01f,
            const std::string assertDetails = "") {
        for (size_t i = 0lu; i < size; i++) {
            ASSERT_NEAR(res[i], ref[i], max_diff) << assertDetails << ", index=" << i;
        }
    }

    template <class T>
    static void compare(std::vector<T> & a, std::vector<T> & b) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(a[i], b[i]);
        }
    }

    void replace(std::string& str, const std::string& from, const std::string& to) {
        std::string::size_type pos = 0;

        while((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    }

    std::string replace(std::string& str, const std::string& from, const int& to) {
        replace(str, from, to_string_c_locale(to));
        return str;
    }

    std::string replace(std::string& str, const std::string& from, const size_t& to) {
        replace(str, from, to_string_c_locale(to));
        return str;
    }

    std::string replace(std::string& str, const std::string& from, const float& to) {
        replace(str, from, to_string_c_locale(to));
        return str;
    }
    // trim from both ends (in place)
    static inline std::string &trim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c){
            return !std::isspace(c);}));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int c){
            return !std::isspace(c);}).base(), s.end());
        return s;
    }

    template <typename T, typename U>
    static inline T div_up(const T a, const U b) {
        assert(b);
        return (a + b - 1) / b;
    }
};

// Check bitness
#include <stdint.h>
#if UINTPTR_MAX == 0xffffffff
    /* 32-bit */
    #define ENVIRONMENT32
#elif UINTPTR_MAX == 0xffffffffffffffff
    /* 64-bit */
    #define ENVIRONMENT64
#else
    # error Unsupported architecture
#endif

/**
 * @brief Splits the RGB channels to either I16 Blob or float blob.
 *
 * The image buffer is assumed to be packed with no support for strides.
 *
 * @param imgBufRGB8 Packed 24bit RGB image (3 bytes per pixel: R-G-B)
 * @param lengthbytesSize Size in bytes of the RGB image. It is equal to amount of pixels times 3 (number of channels)
 * @param input Blob to contain the split image (to 3 channels)
 */
void ConvertImageToInput(unsigned char* imgBufRGB8, size_t lengthbytesSize, InferenceEngine::Blob& input);
