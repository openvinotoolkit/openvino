// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <openvino/openvino.hpp>
#include <samples/slog.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#ifdef USE_OPENCV
const std::unordered_set<std::string> supported_image_extensions =
    {"bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png", "pbm", "pgm", "ppm", "sr", "ras", "tiff", "tif"};
#else
const std::unordered_set<std::string> supported_image_extensions = {"bmp"};
#endif
const std::unordered_set<std::string> supported_numpy_extensions = {"npy"};
const std::unordered_set<std::string> supported_binary_extensions = {"bin"};

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

inline uint64_t get_duration_in_milliseconds(uint64_t duration) {
    return duration * 1000LL;
}

inline uint64_t get_duration_in_nanoseconds(uint64_t duration) {
    return duration * 1000000000LL;
}

inline double get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
};

namespace benchmark_app {
struct InputInfo {
    ov::element::Type type;
    ov::PartialShape partialShape;
    ov::Shape dataShape;
    ov::Layout layout;
    std::vector<float> scale;
    std::vector<float> mean;
    bool is_image() const;
    bool is_image_info() const;
    size_t width() const;
    size_t height() const;
    size_t channels() const;
    size_t batch() const;
    size_t depth() const;
    std::vector<std::string> fileNames;
};
using InputsInfo = std::map<std::string, InputInfo>;
using PartialShapes = std::map<std::string, ngraph::PartialShape>;
}  // namespace benchmark_app

std::vector<std::string> parse_devices(const std::string& device_string);
uint32_t device_default_device_duration_in_seconds(const std::string& device);
std::map<std::string, std::string> parse_value_per_device(const std::vector<std::string>& devices,
                                                          const std::string& values_string);
void parse_value_for_virtual_device(const std::string& device, std::map<std::string, std::string>& values_string);
std::string get_shapes_string(const benchmark_app::PartialShapes& shapes);
size_t get_batch_size(const benchmark_app::InputsInfo& inputs_info);
std::vector<std::string> split(const std::string& s, char delim);
std::map<std::string, std::vector<float>> parse_scale_or_mean(const std::string& scale_mean,
                                                              const benchmark_app::InputsInfo& inputs_info);
std::pair<std::string, std::vector<std::string>> parse_input_files(const std::string& file_paths_string);
std::map<std::string, std::vector<std::string>> parse_input_arguments(const std::vector<std::string>& args);

std::map<std::string, std::vector<std::string>> parse_input_parameters(const std::string& parameter_string,
                                                                       const ov::ParameterVector& input_info);

/// <summary>
/// Parses command line data and data obtained from the function and returns configuration of each input
/// </summary>
/// <param name="shape_string">command-line shape string</param>
/// <param name="layout_string">command-line layout string</param>
/// <param name="batch_size">command-line batch string</param>
/// <param name="tensors_shape_string">command-line data_shape string</param>
/// <param name="scale_string">command-line iscale string</param>
/// <param name="mean_string">command-line imean string</param>
/// <param name="input_info">inputs vector obtained from ov::Model</param>
/// <param name="reshape_required">returns true to this parameter if reshape is required</param>
/// <returns>vector of benchmark_app::InputsInfo elements.
/// Each element is a configuration item for every test configuration case
/// (number of cases is calculated basing on data_shape and other parameters).
/// Each element is a map (input_name, configuration) containing data for each input</returns>
std::vector<benchmark_app::InputsInfo> get_inputs_info(const std::string& shape_string,
                                                       const std::string& layout_string,
                                                       const size_t batch_size,
                                                       const std::string& data_shapes_string,
                                                       const std::map<std::string, std::vector<std::string>>& fileNames,
                                                       const std::string& scale_string,
                                                       const std::string& mean_string,
                                                       const std::vector<ov::Output<const ov::Node>>& input_info,
                                                       bool& reshape_required);

/// <summary>
/// Parses command line data and data obtained from the function and returns configuration of each input
/// </summary>
/// <param name="shape_string">command-line shape string</param>
/// <param name="layout_string">command-line layout string</param>
/// <param name="batch_size">command-line batch string</param>
/// <param name="tensors_shape_string">command-line data_shape string</param>
/// <param name="scale_string">command-line iscale string</param>
/// <param name="mean_string">command-line imean string</param>
/// <param name="input_info">inputs vector obtained from ov::Model</param>
/// <param name="reshape_required">returns true to this parameter if reshape is required</param>
/// <returns>vector of benchmark_app::InputsInfo elements.
/// Each element is a configuration item for every test configuration case
/// (number of cases is calculated basing on data_shape and other parameters).
/// Each element is a map (input_name, configuration) containing data for each
/// input</returns>
std::vector<benchmark_app::InputsInfo> get_inputs_info(const std::string& shape_string,
                                                       const std::string& layout_string,
                                                       const size_t batch_size,
                                                       const std::string& data_shapes_string,
                                                       const std::map<std::string, std::vector<std::string>>& fileNames,
                                                       const std::string& scale_string,
                                                       const std::string& mean_string,
                                                       const std::vector<ov::Output<const ov::Node>>& input_info);

void dump_config(const std::string& filename, const std::map<std::string, ov::AnyMap>& config);
void load_config(const std::string& filename, std::map<std::string, ov::AnyMap>& config);

std::string get_extension(const std::string& name);
bool is_binary_file(const std::string& filePath);
bool is_numpy_file(const std::string& filePath);
bool is_image_file(const std::string& filePath);
bool contains_binaries(const std::vector<std::string>& filePaths);
std::vector<std::string> filter_files_by_extensions(const std::vector<std::string>& filePaths,
                                                    const std::unordered_set<std::string>& extensions);

std::string parameter_name_to_tensor_name(
    const std::string& name,
    const std::vector<ov::Output<const ov::Node>>& inputs_info,
    const std::vector<ov::Output<const ov::Node>>& outputs_info = std::vector<ov::Output<const ov::Node>>());

template <class T>
void convert_io_names_in_map(
    T& map,
    const std::vector<ov::Output<const ov::Node>>& inputs_info,
    const std::vector<ov::Output<const ov::Node>>& outputs_info = std::vector<ov::Output<const ov::Node>>()) {
    T new_map;
    for (auto& item : map) {
        new_map[item.first == "" ? "" : parameter_name_to_tensor_name(item.first, inputs_info, outputs_info)] =
            std::move(item.second);
    }
    map = new_map;
}

void verifyBinaryFile(std::ifstream& binaryFile, const std::string& fileName, const unsigned long inputSize);
template <typename T>
void verifyNumpyFile(std::ifstream& binaryFile,
                     const std::string& fileName,
                     const ov::Shape& inputShape,
                     const unsigned long inputSize) {
    auto fullFileSize = static_cast<std::size_t>(binaryFile.tellg());
    binaryFile.seekg(0, std::ios_base::beg);
    OPENVINO_ASSERT(binaryFile.good(), "Cannot read ", fileName);

    std::string magic_string(6, ' ');
    binaryFile.read(&magic_string[0], magic_string.size());
    OPENVINO_ASSERT(magic_string == "\x93NUMPY",
                    "Numpy file has incorrect magic string. File ",
                    fileName,
                    " might be corrupted");

    binaryFile.ignore(2);
    unsigned short headerSize;
    binaryFile.read((char*)&headerSize, sizeof(headerSize));

    std::string header(headerSize, ' ');
    binaryFile.read(&header[0], header.size());

    int idx, from, to;

    // Verify fortran order is false
    const std::string fortranKey = "'fortran_order':";
    idx = header.find(fortranKey);
    OPENVINO_ASSERT(idx != -1, "Numpy file is missing fortran_order key. File ", fileName, " might be corrupted");
    from = header.find_last_of(' ', idx + fortranKey.size()) + 1;
    to = header.find(',', from);
    auto fortranValue = header.substr(from, to - from);
    OPENVINO_ASSERT(fortranValue == "False", "File ", fileName, " was saved in Fortran order, which is not supported");

    // Verify array shape matches the input's
    const std::string shapeKey = "'shape':";
    idx = header.find(shapeKey);
    OPENVINO_ASSERT(idx != -1, "Numpy file is missing shape key. File ", fileName, " might be corrupted");

    from = header.find('(', idx + shapeKey.size()) + 1;
    to = header.find(')', from);

    std::string shapeStr = header.substr(from, to - from);
    std::vector<size_t> numpyShape;

    if (!shapeStr.empty()) {
        shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), ','), shapeStr.end());

        std::istringstream ss(shapeStr);
        size_t value;
        while (ss >> value) {
            numpyShape.push_back(value);
        }
    }

    if (!inputShape.empty()) {
        OPENVINO_ASSERT(numpyShape.size() == inputShape.size() &&
                            std::equal(numpyShape.begin(), numpyShape.end(), inputShape.begin()),
                        "Numpy array shape mismatch. File ",
                        fileName,
                        " has shape: (",
                        ov::Shape(numpyShape).to_string(),
                        "), expected: (",
                        ov::Shape(numpyShape).to_string(),
                        ")");
    }

    // Verify array data type matches input's
    std::string dataTypeKey = "'descr':";
    idx = header.find(dataTypeKey);
    OPENVINO_ASSERT(idx != -1, "Numpy file is missing descr key. File ", fileName, " might be corrupted");

    from = header.find('\'', idx + dataTypeKey.size()) + 1;
    to = header.find('\'', from);
    std::string dataTypeStr = header.substr(from, to - from);

    if (dataTypeStr.find("<f4") != std::string::npos) {
        auto test = std::is_same<T, float>::value;
        OPENVINO_ASSERT(test,
                        "Numpy array is of 32-bit float format, which does not match input type. File ",
                        fileName);
    } else if (dataTypeStr.find("<f8") != std::string::npos) {
        auto test = std::is_same<T, double>::value;
        OPENVINO_ASSERT(test,
                        "Numpy array is of 64-bit float/double format, which does not match input type. File ",
                        fileName);
    } else if (dataTypeStr.find("<f2") != std::string::npos) {
        auto test = std::is_same<T, short>::value;
        OPENVINO_ASSERT(test,
                        "Numpy array is of 16-bit float/short format, which does not match input type. File ",
                        fileName);
    } else if (dataTypeStr.find("<i4") != std::string::npos) {
        auto test = std::is_same<T, int32_t>::value;
        OPENVINO_ASSERT(test, "Numpy array is of 32-bit int format, which does not match input type. File ", fileName);
    } else if (dataTypeStr.find("<i8") != std::string::npos) {
        auto test = std::is_same<T, int64_t>::value;
        OPENVINO_ASSERT(test, "Numpy array is of 64-bit int format, which does not match input type. File ", fileName);
    } else if (dataTypeStr.find("|u1") != std::string::npos) {
        auto test = std::is_same<T, uint8_t>::value;
        OPENVINO_ASSERT(test, "Numpy array is of 8-bit uint format, which does not match input type. File ", fileName);
    } else {
        throw ov::Exception("Following numpy format is not supported: " + dataTypeStr + ", file: " + fileName);
    }

    auto fileSize = fullFileSize - static_cast<std::size_t>(binaryFile.tellg());
    OPENVINO_ASSERT(fileSize == inputSize,
                    "File ",
                    fileName,
                    " contains ",
                    fileSize,
                    " bytes, but the model expects ",
                    inputSize);
}
