// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file common.hpp
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "slog.hpp"
// clang-format on

// @brief performance counters sort
static constexpr char pcSort[] = "sort";
static constexpr char pcNoSort[] = "no_sort";
static constexpr char pcSimpleSort[] = "simple_sort";

#ifndef UNUSED
#    if defined(_MSC_VER) && !defined(__clang__)
#        define UNUSED
#    else
#        define UNUSED __attribute__((unused))
#    endif
#endif

/**
 * @brief Unicode string wrappers
 */
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#    define tchar                wchar_t
#    define tstring              std::wstring
#    define tmain                wmain
#    define TSTRING2STRING(tstr) wstring2string(tstr)
#else
#    define tchar                char
#    define tstring              std::string
#    define tmain                main
#    define TSTRING2STRING(tstr) tstr
#endif

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)

/**
 * @brief Convert wstring to string
 * @param ref on wstring
 * @return string
 */
inline std::string wstring2string(const std::wstring& wstr) {
    std::string str;
    for (auto&& wc : wstr)
        str += static_cast<char>(wc);
    return str;
}
#endif

/**
 * @brief trim from start (in place)
 * @param s - string to trim
 */
inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) {
                return !std::isspace(c);
            }));
}

/**
 * @brief trim from end (in place)
 * @param s - string to trim
 */
inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(),
                         s.rend(),
                         [](int c) {
                             return !std::isspace(c);
                         })
                .base(),
            s.end());
}

/**
 * @brief trim from both ends (in place)
 * @param s - string to trim
 */
inline std::string& trim(std::string& s) {
    ltrim(s);
    rtrim(s);
    return s;
}
/**
 * @brief Gets filename without extension
 * @param filepath - full file name
 * @return filename without extension
 */
inline std::string fileNameNoExt(const std::string& filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos)
        return filepath;
    return filepath.substr(0, pos);
}

/**
 * @brief Get extension from filename
 * @param filename - name of the file which extension should be extracted
 * @return string with extracted file extension
 */
inline std::string fileExt(const std::string& filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos)
        return "";
    return filename.substr(pos + 1);
}

inline slog::LogStream& operator<<(slog::LogStream& os, const ov::Version& version) {
    os << "Build ................................. ";
    os << version.buildNumber << slog::endl;

    return os;
}

inline slog::LogStream& operator<<(slog::LogStream& os, const std::map<std::string, ov::Version>& versions) {
    for (auto&& version : versions) {
        os << version.first << slog::endl;
        os << version.second << slog::endl;
    }

    return os;
}

/**
 * @brief Writes output data to BMP image
 * @param name - image name
 * @param data - output data
 * @param height - height of the target image
 * @param width - width of the target image
 * @return false if error else true
 */
static UNUSED bool writeOutputBmp(std::string name, unsigned char* data, size_t height, size_t width) {
    std::ofstream outFile;
    outFile.open(name, std::ofstream::binary);
    if (!outFile.is_open()) {
        return false;
    }

    unsigned char file[14] = {
        'B',
        'M',  // magic
        0,
        0,
        0,
        0,  // size in bytes
        0,
        0,  // app data
        0,
        0,  // app data
        40 + 14,
        0,
        0,
        0  // start of data offset
    };
    unsigned char info[40] = {
        40,   0,    0, 0,  // info hd size
        0,    0,    0, 0,  // width
        0,    0,    0, 0,  // height
        1,    0,           // number color planes
        24,   0,           // bits per pixel
        0,    0,    0, 0,  // compression is none
        0,    0,    0, 0,  // image bits size
        0x13, 0x0B, 0, 0,  // horz resolution in pixel / m
        0x13, 0x0B, 0, 0,  // vert resolution (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
        0,    0,    0, 0,  // #colors in palette
        0,    0,    0, 0,  // #important colors
    };

    OPENVINO_ASSERT(
        height < (size_t)std::numeric_limits<int32_t>::max && width < (size_t)std::numeric_limits<int32_t>::max,
        "File size is too big: ",
        height,
        " X ",
        width);

    int padSize = static_cast<int>(4 - (width * 3) % 4) % 4;
    int sizeData = static_cast<int>(width * height * 3 + height * padSize);
    int sizeAll = sizeData + sizeof(file) + sizeof(info);

    file[2] = (unsigned char)(sizeAll);
    file[3] = (unsigned char)(sizeAll >> 8);
    file[4] = (unsigned char)(sizeAll >> 16);
    file[5] = (unsigned char)(sizeAll >> 24);

    info[4] = (unsigned char)(width);
    info[5] = (unsigned char)(width >> 8);
    info[6] = (unsigned char)(width >> 16);
    info[7] = (unsigned char)(width >> 24);

    int32_t negativeHeight = -(int32_t)height;
    info[8] = (unsigned char)(negativeHeight);
    info[9] = (unsigned char)(negativeHeight >> 8);
    info[10] = (unsigned char)(negativeHeight >> 16);
    info[11] = (unsigned char)(negativeHeight >> 24);

    info[20] = (unsigned char)(sizeData);
    info[21] = (unsigned char)(sizeData >> 8);
    info[22] = (unsigned char)(sizeData >> 16);
    info[23] = (unsigned char)(sizeData >> 24);

    outFile.write(reinterpret_cast<char*>(file), sizeof(file));
    outFile.write(reinterpret_cast<char*>(info), sizeof(info));

    unsigned char pad[3] = {0, 0, 0};

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            pixel[0] = data[y * width * 3 + x * 3];
            pixel[1] = data[y * width * 3 + x * 3 + 1];
            pixel[2] = data[y * width * 3 + x * 3 + 2];

            outFile.write(reinterpret_cast<char*>(pixel), 3);
        }
        outFile.write(reinterpret_cast<char*>(pad), padSize);
    }
    return true;
}

/**
 * @brief Adds colored rectangles to the image
 * @param data - data where rectangles are put
 * @param height - height of the rectangle
 * @param width - width of the rectangle
 * @param rectangles - vector points for the rectangle, should be 4x compared to num classes
 * @param classes - vector of classes
 * @param thickness - thickness of a line (in pixels) to be used for bounding boxes
 */
static UNUSED void addRectangles(unsigned char* data,
                                 size_t height,
                                 size_t width,
                                 std::vector<int> rectangles,
                                 std::vector<int> classes,
                                 int thickness) {
    if (height <= 0) {
        throw std::runtime_error("height must be greater than 0");
    }
    if (width <= 0) {
        throw std::runtime_error("width must be greater than 0");
    }
    struct Color {
        unsigned char red;
        unsigned char green;
        unsigned char blue;
    };
    std::vector<Color> colors = {// colors to be used for bounding boxes
                                 {128, 64, 128},  {232, 35, 244}, {70, 70, 70},  {156, 102, 102}, {153, 153, 190},
                                 {153, 153, 153}, {30, 170, 250}, {0, 220, 220}, {35, 142, 107},  {152, 251, 152},
                                 {180, 130, 70},  {60, 20, 220},  {0, 0, 255},   {142, 0, 0},     {70, 0, 0},
                                 {100, 60, 0},    {90, 0, 0},     {230, 0, 0},   {32, 11, 119},   {0, 74, 111},
                                 {81, 0, 81}};

    if (rectangles.size() % 4 != 0 || rectangles.size() / 4 != classes.size()) {
        return;
    }

    for (size_t i = 0; i < classes.size(); i++) {
        int x = rectangles.at(i * 4);
        int y = rectangles.at(i * 4 + 1);
        int w = rectangles.at(i * 4 + 2);
        int h = rectangles.at(i * 4 + 3);

        int cls = classes.at(i) % colors.size();  // color of a bounding box line

        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (w < 0)
            w = 0;
        if (h < 0)
            h = 0;

        if (static_cast<std::size_t>(x) >= width) {
            x = static_cast<int>(width - 1);
            w = 0;
            thickness = 1;
        }
        if (static_cast<std::size_t>(y) >= height) {
            y = static_cast<int>(height - 1);
            h = 0;
            thickness = 1;
        }

        if ((static_cast<std::size_t>(x) + w) >= width) {
            w = static_cast<int>(width - x - 1);
        }
        if ((static_cast<std::size_t>(y) + h) >= height) {
            h = static_cast<int>(height - y - 1);
        }

        thickness = std::min(std::min(thickness, w / 2 + 1), h / 2 + 1);

        size_t shift_first;
        size_t shift_second;
        for (int t = 0; t < thickness; t++) {
            shift_first = (y + t) * width * 3;
            shift_second = (y + h - t) * width * 3;
            for (int ii = x; ii < x + w + 1; ii++) {
                data[shift_first + ii * 3] = colors.at(cls).red;
                data[shift_first + ii * 3 + 1] = colors.at(cls).green;
                data[shift_first + ii * 3 + 2] = colors.at(cls).blue;
                data[shift_second + ii * 3] = colors.at(cls).red;
                data[shift_second + ii * 3 + 1] = colors.at(cls).green;
                data[shift_second + ii * 3 + 2] = colors.at(cls).blue;
            }
        }

        for (int t = 0; t < thickness; t++) {
            shift_first = (x + t) * 3;
            shift_second = (x + w - t) * 3;
            for (int ii = y; ii < y + h + 1; ii++) {
                data[shift_first + ii * width * 3] = colors.at(cls).red;
                data[shift_first + ii * width * 3 + 1] = colors.at(cls).green;
                data[shift_first + ii * width * 3 + 2] = colors.at(cls).blue;
                data[shift_second + ii * width * 3] = colors.at(cls).red;
                data[shift_second + ii * width * 3 + 1] = colors.at(cls).green;
                data[shift_second + ii * width * 3 + 2] = colors.at(cls).blue;
            }
        }
    }
}

inline void showAvailableDevices() {
    ov::Core core;
    std::vector<std::string> devices = core.get_available_devices();

    std::cout << std::endl;
    std::cout << "Available target devices:";
    for (const auto& device : devices) {
        std::cout << "  " << device;
    }
    std::cout << std::endl;
}

inline std::string getFullDeviceName(ov::Core& core, std::string device) {
    try {
        return core.get_property(device, ov::device::full_name);
    } catch (ov::Exception&) {
        return {};
    }
}

static UNUSED void printPerformanceCounts(std::vector<ov::ProfilingInfo> performanceData,
                                          std::ostream& stream,
                                          std::string deviceName,
                                          bool bshowHeader = true,
                                          int precision = 3) {
    std::chrono::microseconds totalTime = std::chrono::microseconds::zero();
    std::chrono::microseconds totalTimeCpu = std::chrono::microseconds::zero();
    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "Performance counts:" << std::endl << std::endl;
    }
    std::ios::fmtflags fmt(std::cout.flags());
    stream << std::fixed << std::setprecision(precision);

    for (const auto& it : performanceData) {
        if (it.real_time.count() > 0) {
            totalTime += it.real_time;
        }
        if (it.cpu_time.count() > 0) {
            totalTimeCpu += it.cpu_time;
        }

        std::string toPrint(it.node_name);
        const int maxPrintLength = 20;

        if (it.node_name.length() >= maxPrintLength) {
            toPrint = it.node_name.substr(0, maxPrintLength - 5);
            toPrint += "...";
        }

        stream << std::setw(maxPrintLength) << std::left << toPrint << " ";
        switch (it.status) {
        case ov::ProfilingInfo::Status::EXECUTED:
            stream << std::setw(21) << std::left << "EXECUTED ";
            break;
        case ov::ProfilingInfo::Status::NOT_RUN:
            stream << std::setw(21) << std::left << "NOT_RUN ";
            break;
        case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
            stream << std::setw(21) << std::left << "OPTIMIZED_OUT ";
            break;
        }

        stream << "layerType: ";
        if (it.node_type.length() >= maxPrintLength) {
            stream << std::setw(maxPrintLength) << std::left << it.node_type.substr(0, maxPrintLength - 3) + "..."
                   << " ";
        } else {
            stream << std::setw(maxPrintLength) << std::left << it.node_type << " ";
        }

        stream << std::setw(30) << std::left << "execType: " + std::string(it.exec_type) << " ";
        stream << "realTime (ms): " << std::setw(10) << std::left << std::fixed << std::setprecision(3)
               << it.real_time.count() / 1000.0 << " ";
        stream << "cpuTime (ms): " << std::setw(10) << std::left << std::fixed << std::setprecision(3)
               << it.cpu_time.count() / 1000.0 << " ";
        stream << std::endl;
    }
    stream << std::setw(25) << std::left << "Total time: " << std::fixed << std::setprecision(3)
           << totalTime.count() / 1000.0 << " milliseconds" << std::endl;
    stream << std::setw(25) << std::left << "Total CPU time: " << std::fixed << std::setprecision(3)
           << totalTimeCpu.count() / 1000.0 << " milliseconds" << std::endl;
    stream << std::endl;
    stream << "Full device name: " << deviceName << std::endl;
    stream << std::endl;
    stream.flags(fmt);
}

static inline std::string double_to_string(const double number) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << number;
    return ss.str();
}

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
static inline void fill_random(ov::Tensor& tensor,
                               T rand_min = std::numeric_limits<uint8_t>::min(),
                               T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    size_t tensor_size = tensor.get_size();
    if (0 == tensor_size) {
        throw std::runtime_error(
            "Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    }
    T* data = tensor.data<T>();
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = static_cast<T>(distribution(gen));
    }
}

static inline void fill_tensor_random(ov::Tensor tensor) {
    switch (tensor.get_element_type()) {
    case ov::element::f32:
        fill_random<float, float>(tensor);
        break;
    case ov::element::f64:
        fill_random<double, double>(tensor);
        break;
    case ov::element::f16:
        fill_random<short, short>(tensor);
        break;
    case ov::element::i32:
        fill_random<int32_t, int32_t>(tensor);
        break;
    case ov::element::i64:
        fill_random<int64_t, int64_t>(tensor);
        break;
    case ov::element::u8:
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        fill_random<uint8_t, uint32_t>(tensor);
        break;
    case ov::element::i8:
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        fill_random<int8_t, int32_t>(tensor, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
        break;
    case ov::element::u16:
        fill_random<uint16_t, uint16_t>(tensor);
        break;
    case ov::element::i16:
        fill_random<int16_t, int16_t>(tensor);
        break;
    case ov::element::boolean:
        fill_random<uint8_t, uint32_t>(tensor, 0, 1);
        break;
    default:
        OPENVINO_THROW("Input type is not supported for a tensor");
    }
}

static UNUSED bool sort_pc_descend(const ov::ProfilingInfo& profiling1, const ov::ProfilingInfo& profiling2) {
    return profiling1.real_time > profiling2.real_time;
}

static UNUSED void printPerformanceCountsSort(std::vector<ov::ProfilingInfo> performanceData,
                                              std::ostream& stream,
                                              std::string deviceName,
                                              std::string sorttype,
                                              bool bshowHeader = true,
                                              int precision = 3) {
    std::chrono::microseconds totalTime = std::chrono::microseconds::zero();
    std::chrono::microseconds totalTimeCpu = std::chrono::microseconds::zero();

    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "Performance counts:" << std::endl << std::endl;
    }
    std::ios::fmtflags fmt(std::cout.flags());
    stream << std::fixed << std::setprecision(precision);

    for (const auto& it : performanceData) {
        if (it.real_time.count() > 0) {
            totalTime += it.real_time;
        }
        if (it.cpu_time.count() > 0) {
            totalTimeCpu += it.cpu_time;
        }
    }
    if (totalTime.count() != 0) {
        std::vector<ov::ProfilingInfo> sortPerfCounts{std::begin(performanceData), std::end(performanceData)};
        if (sorttype == pcSort || sorttype == pcSimpleSort) {
            std::sort(sortPerfCounts.begin(), sortPerfCounts.end(), sort_pc_descend);
        }

        for (const auto& it : sortPerfCounts) {
            if ((sorttype == pcSimpleSort && it.status == ov::ProfilingInfo::Status::EXECUTED) ||
                sorttype != pcSimpleSort) {
                std::string toPrint(it.node_name);
                const int maxPrintLength = 20;

                if (it.node_name.length() >= maxPrintLength) {
                    toPrint = it.node_name.substr(0, maxPrintLength - 5);
                    toPrint += "...";
                }

                stream << std::setw(maxPrintLength) << std::left << toPrint << " ";
                switch (it.status) {
                case ov::ProfilingInfo::Status::EXECUTED:
                    stream << std::setw(21) << std::left << "EXECUTED ";
                    break;
                case ov::ProfilingInfo::Status::NOT_RUN:
                    stream << std::setw(21) << std::left << "NOT_RUN ";
                    break;
                case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
                    stream << std::setw(21) << std::left << "OPTIMIZED_OUT ";
                    break;
                }

                stream << "layerType: ";
                if (it.node_type.length() >= maxPrintLength) {
                    stream << std::setw(maxPrintLength) << std::left
                           << it.node_type.substr(0, maxPrintLength - 3) + "..."
                           << " ";
                } else {
                    stream << std::setw(maxPrintLength) << std::left << it.node_type << " ";
                }

                stream << std::setw(30) << std::left << "execType: " + std::string(it.exec_type) << " ";
                stream << "realTime (ms): " << std::setw(10) << std::left << std::fixed << std::setprecision(3)
                       << it.real_time.count() / 1000.0 << " ";
                stream << "cpuTime (ms): " << std::setw(10) << std::left << std::fixed << std::setprecision(3)
                       << it.cpu_time.count() / 1000.0 << " ";

                double opt_proportion = it.real_time.count() * 100.0 / totalTime.count();
                std::stringstream opt_proportion_ss;
                opt_proportion_ss << std::fixed << std::setprecision(2) << opt_proportion;
                std::string opt_proportion_str = opt_proportion_ss.str();
                if (opt_proportion_str == "0.00") {
                    opt_proportion_str = "N/A";
                }
                stream << std::setw(20) << std::left << "proportion: " + opt_proportion_str + "%";
                stream << std::endl;
            }
        }
    }
    stream << std::setw(25) << std::left << "Total time: " + std::to_string(totalTime.count() / 1000.0)
           << " milliseconds" << std::endl;
    stream << std::endl;
    stream << "Full device name: " << deviceName << std::endl;
    stream << std::endl;
    stream.flags(fmt);
}
