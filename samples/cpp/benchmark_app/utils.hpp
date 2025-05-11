// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <iomanip>
#include <map>
#include <openvino/openvino.hpp>
#include <samples/slog.hpp>
#include <string>
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
using PartialShapes = std::map<std::string, ov::PartialShape>;
}  // namespace benchmark_app

bool can_measure_as_static(const std::vector<benchmark_app::InputsInfo>& app_input_info);
bool is_virtual_device(const std::string& device_name);
bool is_virtual_device_found(const std::vector<std::string>& device_names);
void update_device_properties_setting(const std::string& device_name,
                                      ov::AnyMap& config,
                                      std::pair<std::string, ov::Any> device_property);
std::vector<std::string> parse_devices(const std::string& device_string);
uint32_t device_default_device_duration_in_seconds(const std::string& device);
std::map<std::string, std::string> parse_value_per_device(const std::vector<std::string>& devices,
                                                          const std::string& values_string);
void parse_value_for_virtual_device(const std::string& device, std::map<std::string, std::string>& values_string);
template <typename T>
void update_device_config_for_virtual_device(const std::string& value,
                                             ov::AnyMap& device_config,
                                             ov::Property<T, ov::PropertyMutability::RW> property);
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
