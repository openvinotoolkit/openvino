// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"

#include "gtest/gtest.h"
#include "openvino/core/except.hpp"
#include "precomp.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    define _WINSOCKAPI_

#    include <windows.h>

#    include "psapi.h"
#endif

namespace ov {
namespace test {
namespace utils {

std::ostream& operator<<(std::ostream& os, OpType type) {
    switch (type) {
    case OpType::SCALAR:
        os << "SCALAR";
        break;
    case OpType::VECTOR:
        os << "VECTOR";
        break;
    default:
        OPENVINO_THROW("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::string generateTestFilePrefix() {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string testName = testInfo->test_case_name();
    testName += testInfo->name();
    testName = std::to_string(std::hash<std::string>()(testName));
    std::stringstream ss;
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch());
    ss << testName << "_" << std::this_thread::get_id() << "_" << ts.count();
    testName = ss.str();
    std::replace(testName.begin(), testName.end(), ':', '_');
    return testName;
}

#ifdef _WIN32
static PROCESS_MEMORY_COUNTERS getMemoryInfo() {
    static PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb))
        throw std::runtime_error("Can't get system memory values");
    return pmc;
}

size_t getVmSizeInKB() {
    return getMemoryInfo().PagefileUsage / 1024;
}

size_t getVmRSSInKB() {
    return getMemoryInfo().WorkingSetSize / 1024;
}

#else

size_t getSystemDataByName(char* name) {
    auto parseLine = [](std::string line) -> size_t {
        std::string res = "";
        for (auto c : line)
            if (isdigit(c))
                res += c;
        if (res.empty())
            throw std::runtime_error("Can't get system memory values");
        return std::stoul(res);
    };

    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    bool status = false;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, name, strlen(name)) == 0) {
                result = parseLine(line);
                status = true;
                break;
            }
        }
        fclose(file);
    }
    if (!status)
        throw std::runtime_error("Can't get system memory values");
    return result;
}

size_t getVmSizeInKB() {
    return getSystemDataByName(const_cast<char*>("VmSize:"));
}

size_t getVmRSSInKB() {
    return getSystemDataByName(const_cast<char*>("VmRSS:"));
}

#endif

std::vector<uint8_t> color_test_image(size_t height, size_t width, int b_step, ov::preprocess::ColorFormat format) {
    // Test all possible r/g/b values within dimensions
    int b_dim = 255 / b_step + 1;
    auto input_yuv = std::vector<uint8_t>(height * b_dim * width * 3 / 2);
    for (int b = 0; b <= 255; b += b_step) {
        for (size_t y = 0; y < height / 2; y++) {
            for (size_t x = 0; x < width / 2; x++) {
                int r = static_cast<int>(y) * 512 / static_cast<int>(height);
                int g = static_cast<int>(x) * 512 / static_cast<int>(width);
                // Can't use random y/u/v for testing as this can lead to invalid R/G/B values
                int y_val = ((66 * r + 129 * g + 25 * b + 128) / 256) + 16;
                int u_val = ((-38 * r - 74 * g + 112 * b + 128) / 256) + 128;
                int v_val = ((112 * r - 94 * g + 18 * b + 128) / 256) + 128;

                size_t b_offset = height * width * b / b_step * 3 / 2;
                if (ov::preprocess::ColorFormat::I420_SINGLE_PLANE == format ||
                    ov::preprocess::ColorFormat::I420_THREE_PLANES == format) {
                    size_t u_index = b_offset + height * width + y * width / 2 + x;
                    size_t v_index = u_index + height * width / 4;
                    input_yuv[u_index] = u_val;
                    input_yuv[v_index] = v_val;
                } else {
                    size_t uv_index = b_offset + height * width + y * width + x * 2;
                    input_yuv[uv_index] = u_val;
                    input_yuv[uv_index + 1] = v_val;
                }
                size_t y_index = b_offset + y * 2 * width + x * 2;
                input_yuv[y_index] = y_val;
                input_yuv[y_index + 1] = y_val;
                input_yuv[y_index + width] = y_val;
                input_yuv[y_index + width + 1] = y_val;
            }
        }
    }
    return input_yuv;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
