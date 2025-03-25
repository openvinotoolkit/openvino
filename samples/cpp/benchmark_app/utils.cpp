// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>

#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "utils.hpp"
// clang-format on

#ifdef JSON_HEADER
#    include <json.hpp>
#else
#    include <nlohmann/json.hpp>
#endif

#ifdef USE_OPENCV
#    include <opencv2/core.hpp>
#endif

namespace benchmark_app {
bool InputInfo::is_image() const {
    if ((layout != "NCHW") && (layout != "NHWC") && (layout != "CHW") && (layout != "HWC"))
        return false;
    // If data_shape is still empty, assume this is still an Image and tensor shape will be filled later
    return (dataShape.empty() || channels() == 3);
}
bool InputInfo::is_image_info() const {
    if (layout != "NC")
        return false;
    return (channels() >= 2);
}
size_t InputInfo::width() const {
    return dataShape.at(ov::layout::width_idx(layout));
}
size_t InputInfo::height() const {
    return dataShape.at(ov::layout::height_idx(layout));
}
size_t InputInfo::channels() const {
    return dataShape.at(ov::layout::channels_idx(layout));
}
size_t InputInfo::batch() const {
    return dataShape.at(ov::layout::batch_idx(layout));
}
size_t InputInfo::depth() const {
    return dataShape.at(ov::layout::depth_idx(layout));
}
}  // namespace benchmark_app

uint32_t device_default_device_duration_in_seconds(const std::string& device) {
    static const std::map<std::string, uint32_t> deviceDefaultDurationInSeconds{{"CPU", 60},
                                                                                {"GPU", 60},
                                                                                {"NPU", 60},
                                                                                {"UNKNOWN", 120}};
    uint32_t duration = 0;
    for (const auto& deviceDurationInSeconds : deviceDefaultDurationInSeconds) {
        if (device.find(deviceDurationInSeconds.first) != std::string::npos) {
            duration = std::max(duration, deviceDurationInSeconds.second);
        }
    }
    if (duration == 0) {
        const auto unknownDeviceIt = find_if(deviceDefaultDurationInSeconds.begin(),
                                             deviceDefaultDurationInSeconds.end(),
                                             [](std::pair<std::string, uint32_t> deviceDuration) {
                                                 return deviceDuration.first == "UNKNOWN";
                                             });

        if (unknownDeviceIt == deviceDefaultDurationInSeconds.end()) {
            throw std::logic_error("UNKNOWN device was not found in the device duration list");
        }
        duration = unknownDeviceIt->second;
        slog::warn << "Default duration " << duration << " seconds for unknown device '" << device << "' is used"
                   << slog::endl;
    }
    return duration;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<float> split_float(const std::string& s, char delim) {
    std::vector<float> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(std::stof(item));
    }
    return result;
}

bool can_measure_as_static(const std::vector<benchmark_app::InputsInfo>& app_input_info) {
    for (const benchmark_app::InputsInfo& info : app_input_info) {
        for (const auto& pair : info) {
            if (pair.second.partialShape.is_dynamic() && app_input_info.size() > 1) {
                return false;
            }
        }
    }
    return true;
}

static const std::vector<std::string> meta_plugins{"MULTI", "HETERO", "AUTO"};
bool is_virtual_device(const std::string& device_name) {
    return std::find(meta_plugins.begin(), meta_plugins.end(), device_name) != meta_plugins.end();
}

bool is_virtual_device_found(const std::vector<std::string>& device_names) {
    for (const auto& device_name : device_names) {
        if (is_virtual_device(device_name)) {
            return true;
        }
    }
    return false;
}

void update_device_properties_setting(const std::string& device_name,
                                      ov::AnyMap& config,
                                      std::pair<std::string, ov::Any> device_property) {
    // overriding if property {key, value} is already existed in config["DEVICE_PROPERTIES"][device_name],
    // if not, insert this {key, value} into config["DEVICE_PROPERTIES"][device_name].

    // check and create property {"DEVICE_PROPERTIES": ov::AnyMap{hw_device, ov::AnyMap{}}} if not exist in config
    if (config.find(ov::device::properties.name()) == config.end()) {
        config[ov::device::properties.name()] = ov::AnyMap{};
        config[ov::device::properties.name()].as<ov::AnyMap>().insert({device_name, ov::AnyMap{device_property}});
        return;
    }

    auto& device_properties = config[ov::device::properties.name()].as<ov::AnyMap>();
    if (device_properties.find(device_name) == device_properties.end()) {
        device_properties.insert({device_name, ov::AnyMap{device_property}});
        return;
    }

    auto& secondary_property = device_properties[device_name].as<ov::AnyMap>();
    // overwrite if this config existed
    secondary_property.erase(device_property.first);
    secondary_property.insert(device_property);
}

std::vector<std::string> parse_devices(const std::string& device_string) {
    std::string comma_separated_devices = device_string;
    auto colon = comma_separated_devices.find(":");
    std::vector<std::string> result;
    if (colon != std::string::npos) {
        auto target_device = comma_separated_devices.substr(0, colon);
        if (is_virtual_device(target_device)) {
            result.push_back(target_device);
        }
        auto bracket = comma_separated_devices.find("(");  // e.g. in BATCH:GPU(4)
        comma_separated_devices = comma_separated_devices.substr(colon + 1, bracket - colon - 1);
    }

    auto devices = split(comma_separated_devices, ',');
    for (auto&& device : devices) {
        // e.g. in AUTO:-CPU,-GPU
        if (device.front() == '-')
            device.erase(device.begin());
        result.push_back(device);
    }
    return result;
}

void parse_value_for_virtual_device(const std::string& device, std::map<std::string, std::string>& values_string) {
    auto item_virtual = values_string.find(device);
    if (item_virtual != values_string.end() && values_string.size() > 1) {
        if (device == "MULTI") {
            // Remove the element that the key is virtual device MULTI
            // e.g. MULTI:xxx -nstreams 2 will set nstreams 2 to xxx.
            values_string.erase(item_virtual);
        } else if ((device == "AUTO") || (device == "HETERO")) {
            // Just keep the element that the key is virtual device AUTO/HETERO
            // e.g. AUTO:xxx,xxx -nstreams 2 will trigger exception that AUTO plugin didn't support nstream property.
            auto value = item_virtual->second;
            values_string.clear();
            values_string[device] = value;
            return;
        }
    }
    std::stringstream ss;
    auto iter = values_string.begin();
    while (iter != values_string.end()) {
        if (iter->first == device) {
            iter++;
            continue;
        }
        if (ss.str().empty())
            ss << '{';
        else
            ss << ',';
        ss << iter->first << ":" << iter->second;
        iter = values_string.erase(iter);
    }
    if (!ss.str().empty()) {
        ss << '}';
        values_string[device] = ss.str();
    }
    return;
}

template <typename T>
void update_device_config_for_virtual_device(const std::string& value,
                                             ov::AnyMap& device_config,
                                             ov::Property<T, ov::PropertyMutability::RW> property) {
    // check if the element contains the hardware device property
    if (split(value, ':').size() == 1) {
        device_config[property.name()] = value;
    } else {
        // set device nstreams properties in the AUTO/MULTI/HETERO plugin
        std::stringstream strm(value);
        std::map<std::string, std::string> devices_property;
        ov::util::Read<std::map<std::string, std::string>>{}(strm, devices_property);
        for (const auto& it : devices_property) {
            const auto& device_name = it.first;
            const auto& device_value = it.second;
            update_device_properties_setting(device_name, device_config, property(device_value));
        }
    }
}

void update_device_config_for_virtual_device(const std::string& value,
                                             ov::AnyMap& device_config,
                                             ov::Property<ov::streams::Num, ov::PropertyMutability::RW> property) {
    return update_device_config_for_virtual_device<ov::streams::Num>(value, device_config, property);
}

void update_device_config_for_virtual_device(const std::string& value,
                                             ov::AnyMap& device_config,
                                             ov::Property<ov::element::Type, ov::PropertyMutability::RW> property) {
    return update_device_config_for_virtual_device<ov::element::Type>(value, device_config, property);
}

std::map<std::string, std::string> parse_value_per_device(const std::vector<std::string>& devices,
                                                          const std::string& values_string) {
    //  Format: <device1>:<value1>,<device2>:<value2> or just <value>
    std::map<std::string, std::string> result;
    auto device_value_strings = split(values_string, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec = split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto device_name = device_value_vec.at(0);
            auto value = device_value_vec.at(1);
            auto it = std::find(devices.begin(), devices.end(), device_name);
            if (it != devices.end()) {
                result[device_name] = value;
            } else {
                std::string devices_list = "";
                for (auto& device : devices)
                    devices_list += device + " ";
                devices_list.pop_back();
                throw std::logic_error("Failed to set property to '" + device_name +
                                       "' which is not found whthin the target devices list '" + devices_list + "'!");
            }
        } else if (device_value_vec.size() == 1) {
            auto value = device_value_vec.at(0);
            for (auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}

size_t get_batch_size(const benchmark_app::InputsInfo& inputs_info) {
    size_t batch_size = 0;
    for (auto& info : inputs_info) {
        if (ov::layout::has_batch(info.second.layout)) {
            if (batch_size == 0)
                batch_size = info.second.batch();
            else if (batch_size != info.second.batch())
                throw std::logic_error("Can't deterimine batch size: batch is "
                                       "different for different inputs!");
        }
    }
    if (batch_size == 0) {
        batch_size = 1;
    }
    return batch_size;
}

std::string get_shapes_string(const benchmark_app::PartialShapes& shapes) {
    std::stringstream ss;
    for (auto& shape : shapes) {
        if (!ss.str().empty())
            ss << ", ";
        ss << "\'" << shape.first << "': " << shape.second;
    }
    return ss.str();
}

std::map<std::string, std::vector<float>> parse_scale_or_mean(const std::string& scale_mean,
                                                              const benchmark_app::InputsInfo& inputs_info) {
    //  Format: data[255,255,255],info[255,255,255]
    std::map<std::string, std::vector<float>> return_value;

    std::string search_string = scale_mean;
    auto start_pos = search_string.find_first_of('[');
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        auto input_name = search_string.substr(0, start_pos);
        auto input_value_string = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        auto input_value = split_float(input_value_string, ',');

        if (!input_name.empty()) {
            if (inputs_info.count(input_name)) {
                return_value[input_name] = input_value;
            }
            // ignore wrong input name
        } else {
            for (auto& item : inputs_info) {
                if (item.second.is_image())
                    return_value[item.first] = input_value;
            }
            search_string.clear();
            break;
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',')
            break;
        search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + scale_mean);
    return return_value;
}

std::pair<std::string, std::vector<std::string>> parse_input_files(const std::string& file_paths_string) {
    auto search_string = file_paths_string;
    std::string input_name = "";
    std::vector<std::string> file_paths;

    // parse strings like <input1>:file1,file2,file3 and get name from them
    size_t semicolon_pos = search_string.find_first_of(":");
    size_t quote_pos = search_string.find_first_of("\"");
    if (semicolon_pos != std::string::npos && quote_pos != std::string::npos && semicolon_pos > quote_pos) {
        // if : is found after opening " symbol - this means that " belongs to pathname
        semicolon_pos = std::string::npos;
    }
    if (search_string.length() > 2 && semicolon_pos == 1 && search_string[2] == '\\') {
        // Special case like C:\ denotes drive name, not an input name
        semicolon_pos = std::string::npos;
    }

    if (semicolon_pos != std::string::npos) {
        input_name = search_string.substr(0, semicolon_pos);
        search_string = search_string.substr(semicolon_pos + 1);
    }

    // parse file1,file2,file3 and get vector of paths
    size_t coma_pos = 0;
    do {
        coma_pos = search_string.find_first_of(',');
        file_paths.push_back(search_string.substr(0, coma_pos));
        if (coma_pos == std::string::npos) {
            search_string = "";
            break;
        }
        search_string = search_string.substr(coma_pos + 1);
    } while (coma_pos != std::string::npos);

    if (!search_string.empty())
        throw std::logic_error("Can't parse file paths for input " + input_name +
                               " in input parameter string: " + file_paths_string);

    return {input_name, file_paths};
}

std::map<std::string, std::vector<std::string>> parse_input_arguments(const std::vector<std::string>& args) {
    std::map<std::string, std::vector<std::string>> mapped_files = {};
    auto args_it = begin(args);
    const auto is_image_arg = [](const std::string& s) {
        return s == "-i";
    };
    const auto is_arg = [](const std::string& s) {
        return s.front() == '-';
    };
    while (args_it != args.end()) {
        const auto files_start = std::find_if(args_it, end(args), is_image_arg);
        if (files_start == end(args)) {
            break;
        }
        const auto files_begin = std::next(files_start);
        const auto files_end = std::find_if(files_begin, end(args), is_arg);
        for (auto f = files_begin; f != files_end; ++f) {
            auto files = parse_input_files(*f);
            if (mapped_files.find(files.first) == mapped_files.end()) {
                mapped_files[files.first] = {};
            }

            for (auto& file : files.second) {
                if (file == "image_info" || file == "random") {
                    mapped_files[files.first].push_back(file);
                } else {
                    readInputFilesArguments(mapped_files[files.first], file);
                }
            }
        }
        args_it = files_end;
    }
    size_t max_files = 20;
    for (auto& files : mapped_files) {
        if (files.second.size() <= max_files) {
            slog::info << "For input " << files.first << " " << files.second.size() << " files were added. "
                       << slog::endl;
        } else {
            slog::info << "For input " << files.first << " " << files.second.size() << " files were added. "
                       << " The number of files will be limited to " << max_files << "." << slog::endl;
            files.second.resize(20);
        }
    }

    return mapped_files;
}

std::map<std::string, std::vector<std::string>> parse_input_parameters(
    const std::string& parameter_string,
    const std::vector<ov::Output<const ov::Node>>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::vector<std::string>> return_value;
    std::string search_string = parameter_string;
    auto start_pos = search_string.find_first_of('[');
    auto input_name = search_string.substr(0, start_pos);
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        if (start_pos)
            input_name = search_string.substr(0, start_pos);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[parameter_name_to_tensor_name(input_name, input_info)].push_back(input_value);
        } else {
            for (auto& item : input_info) {
                return_value[item.get_any_name()].push_back(input_value);
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || (search_string.front() != ',' && search_string.front() != '['))
            break;
        if (search_string.front() == ',')
            search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + parameter_string);
    return return_value;
}

std::vector<benchmark_app::InputsInfo> get_inputs_info(const std::string& shape_string,
                                                       const std::string& layout_string,
                                                       const size_t batch_size,
                                                       const std::string& data_shapes_string,
                                                       const std::map<std::string, std::vector<std::string>>& fileNames,
                                                       const std::string& scale_string,
                                                       const std::string& mean_string,
                                                       const std::vector<ov::Output<const ov::Node>>& input_info,
                                                       bool& reshape_required) {
    std::map<std::string, std::vector<std::string>> shape_map = parse_input_parameters(shape_string, input_info);
    std::map<std::string, std::vector<std::string>> data_shapes_map =
        parse_input_parameters(data_shapes_string, input_info);
    std::map<std::string, std::vector<std::string>> layout_map = parse_input_parameters(layout_string, input_info);

    size_t min_size = 1, max_size = 1;
    if (!data_shapes_map.empty()) {
        min_size = std::min_element(data_shapes_map.begin(),
                                    data_shapes_map.end(),
                                    [](std::pair<std::string, std::vector<std::string>> a,
                                       std::pair<std::string, std::vector<std::string>> b) {
                                        return a.second.size() < b.second.size() && a.second.size() != 1;
                                    })
                       ->second.size();

        max_size = std::max_element(data_shapes_map.begin(),
                                    data_shapes_map.end(),
                                    [](std::pair<std::string, std::vector<std::string>> a,
                                       std::pair<std::string, std::vector<std::string>> b) {
                                        return a.second.size() < b.second.size();
                                    })
                       ->second.size();
        if (min_size != max_size) {
            throw std::logic_error(
                "Shapes number for every input should be either 1 or should be equal to shapes number of other inputs");
        }
        slog::info << "Number of test configurations is calculated basing on -data_shape parameter" << slog::endl;
    } else if (fileNames.size() > 0) {
        slog::info << "Number of test configurations is calculated basing on number of input images" << slog::endl;
        min_size = std::min_element(fileNames.begin(),
                                    fileNames.end(),
                                    [](std::pair<std::string, std::vector<std::string>> a,
                                       std::pair<std::string, std::vector<std::string>> b) {
                                        return a.second.size() < b.second.size() && a.second.size() != 1;
                                    })
                       ->second.size();

        max_size = std::max_element(fileNames.begin(),
                                    fileNames.end(),
                                    [](std::pair<std::string, std::vector<std::string>> a,
                                       std::pair<std::string, std::vector<std::string>> b) {
                                        return a.second.size() < b.second.size();
                                    })
                       ->second.size();
        if (min_size != max_size) {
            slog::warn << "Number of input files is different for some inputs, minimal number of files will be used ("
                       << min_size << ")" << slog::endl;
        }
    }

    reshape_required = false;

    std::map<std::string, int> currentFileCounters;
    for (auto& item : input_info) {
        currentFileCounters[item.get_any_name()] = 0;
    }

    std::vector<benchmark_app::InputsInfo> info_maps;
    for (size_t input_id = 0; input_id < min_size; ++input_id) {
        benchmark_app::InputsInfo info_map;

        bool is_there_at_least_one_batch_dim = false;
        for (auto& item : input_info) {
            benchmark_app::InputInfo info;
            auto name = item.get_any_name();

            // Layout
            if (layout_map.count(name)) {
                if (layout_map.at(name).size() > 1) {
                    throw std::logic_error(
                        "layout command line parameter doesn't support multiple layouts for one input.");
                }
                info.layout = ov::Layout(layout_map.at(name)[0]);
            } else {
                info.layout = dynamic_cast<const ov::op::v0::Parameter&>(*item.get_node()).get_layout();
            }

            // Calculating default layout values if needed
            std::string newLayout = "";
            if (info.layout.empty()) {
                switch (item.get_partial_shape().size()) {
                case 3:
                    newLayout = (item.get_partial_shape()[2].get_max_length() <= 4 &&
                                 item.get_partial_shape()[0].get_max_length() > 4)
                                    ? "HWC"
                                    : "CHW";
                    break;
                case 4:
                    // Rough check for layout type, basing on max number of image channels
                    newLayout = (item.get_partial_shape()[3].get_max_length() <= 4 &&
                                 item.get_partial_shape()[1].get_max_length() > 4)
                                    ? "NHWC"
                                    : "NCHW";
                    break;
                }
                if (newLayout != "") {
                    info.layout = ov::Layout(newLayout);
                }
                if (info_maps.empty()) {  // Show warnings only for 1st test case config, as for other test cases
                                          // they will be the same
                    slog::warn << item.get_any_name() << ": layout is not set explicitly"
                               << (newLayout != "" ? std::string(", so it is defaulted to ") + newLayout : "")
                               << ". It is STRONGLY recommended to set layout manually to avoid further issues."
                               << slog::endl;
                }
            }

            // Precision
            info.type = item.get_element_type();
            // Partial Shape
            if (shape_map.count(name)) {
                if (shape_map.at(name).size() > 1) {
                    throw std::logic_error(
                        "shape command line parameter doesn't support multiple shapes for one input.");
                }
                info.partialShape = shape_map.at(name)[0];
                reshape_required = true;
            } else {
                info.partialShape = item.get_partial_shape();
            }

            // Files might be mapped without input name. In case of only one input we may map them to the only input
            // directly
            std::string filesInputName =
                fileNames.size() == 1 && input_info.size() == 1 && fileNames.begin()->first == "" ? "" : name;

            // Tensor Shape
            if (info.partialShape.is_dynamic() && data_shapes_map.count(name)) {
                ov::PartialShape p_shape = data_shapes_map.at(name)[input_id % data_shapes_map.at(name).size()];
                if (p_shape.is_dynamic()) {
                    throw std::logic_error("Data shape always should be static, " + p_shape.to_string() +
                                           " is dynamic.");
                }
                if (info.partialShape.compatible(p_shape)) {
                    info.dataShape = p_shape.to_shape();
                } else {
                    throw std::logic_error("Data shape " + p_shape.to_string() + "provided for input " + name +
                                           "is not compatible with partial shape " + info.partialShape.to_string() +
                                           " for this input.");
                }
            } else if (info.partialShape.is_dynamic() && fileNames.count(filesInputName) && info.is_image()) {
                auto& namesVector = fileNames.at(filesInputName);
                if (contains_binaries(namesVector)) {
                    throw std::logic_error("Input files list for input " + item.get_any_name() +
                                           " contains binary file(s) and input shape is dynamic. Tensor shape should "
                                           "be defined explicitly (using -data_shape).");
                }

                info.dataShape = ov::Shape(info.partialShape.size(), 0);
                for (size_t i = 0; i < info.partialShape.size(); i++) {
                    auto& dim = info.partialShape[i];
                    if (dim.is_static()) {
                        info.dataShape[i] = dim.get_length();
                    }
                }

                size_t tensorBatchSize = std::max(batch_size, (size_t)1);
                if (ov::layout::has_batch(info.layout)) {
                    if (info.batch()) {
                        tensorBatchSize = std::max(tensorBatchSize, info.batch());
                    } else {
                        info.dataShape[ov::layout::batch_idx(info.layout)] = tensorBatchSize;
                    }
                }

                size_t h = 0;
                size_t w = 0;
                std::vector<size_t> shape;
                size_t fileIdx = currentFileCounters[item.get_any_name()];
                for (; fileIdx < currentFileCounters[item.get_any_name()] + tensorBatchSize; fileIdx++) {
                    if (fileIdx >= namesVector.size()) {
                        throw std::logic_error(
                            "Not enough files to fill in full batch (number of files should be a multiple of batch "
                            "size if -data_shape parameter is omitted and shape is dynamic)");
                    }
                    FormatReader::ReaderPtr reader(namesVector[fileIdx].c_str());
                    if ((w && w != reader->width()) || (h && h != reader->height()) ||
                        (!shape.empty() && shape != reader->shape())) {
                        throw std::logic_error(
                            "File dimensions putting into one batch should be of the same dimensionality if input "
                            "shape is dynamic and -data_shape is omitted. Problem file: " +
                            namesVector[fileIdx]);
                    }
                    h = reader->height();
                    w = reader->width();
                    shape = reader->shape();
                }
                currentFileCounters[item.get_any_name()] = fileIdx;
                if (shape.size() == 2) {  // Has only h and w
                    if (!info.dataShape[ov::layout::height_idx(info.layout)]) {
                        info.dataShape[ov::layout::height_idx(info.layout)] = h;
                    }
                    if (!info.dataShape[ov::layout::width_idx(info.layout)]) {
                        info.dataShape[ov::layout::width_idx(info.layout)] = w;
                    }
                } else {  // Is numpy array
                    size_t shape_idx = 0;
                    if (info.dataShape.size() != shape.size()) {
                        throw std::logic_error("Shape required by the input and file shape do not have the same rank. "
                                               "Input: " +
                                               item.get_any_name() + ", File name: " + namesVector[fileIdx - 1]);
                    }
                    for (size_t i = ov::layout::batch_idx(info.layout);
                         i < ov::layout::batch_idx(info.layout) + info.dataShape.size();
                         ++i) {
                        if (!info.dataShape[i]) {
                            info.dataShape[i] = shape.at(shape_idx);
                        }
                        shape_idx++;
                    }
                }

                if (std::any_of(info.dataShape.begin(), info.dataShape.end(), [](size_t d) {
                        return d == 0;
                    })) {
                    throw std::logic_error("Not enough information in shape and file to determine tensor shape "
                                           "autmatically. Input: " +
                                           item.get_any_name() + ", File name: " + namesVector[fileIdx - 1]);
                }

            } else if (info.partialShape.is_static()) {
                info.dataShape = info.partialShape.get_shape();
                if (data_shapes_map.find(name) != data_shapes_map.end()) {
                    throw std::logic_error(
                        "Model's input \"" + name +
                        "\" is static. Use -shape argument for static inputs instead of -data_shape.");
                }
            } else if (!data_shapes_map.empty()) {
                throw std::logic_error("Can't find model input name \"" + name + "\" in \"-data_shape " +
                                       data_shapes_string + "\" command line parameter");
            } else {
                throw std::logic_error("-i or -data_shape command line parameter should be set for all inputs in case "
                                       "of model with dynamic shapes.");
            }

            // Update shape with batch if needed (only in static shape case)
            // Update blob shape only not affecting network shape to trigger dynamic batch size case
            if (batch_size != 0) {
                if (ov::layout::has_batch(info.layout)) {
                    std::size_t batch_index = ov::layout::batch_idx(info.layout);
                    if (info.dataShape.at(batch_index) != batch_size) {
                        if (info.partialShape.is_static()) {
                            info.partialShape[batch_index] = batch_size;
                        }
                        info.dataShape[batch_index] = batch_size;
                        reshape_required = true;
                        is_there_at_least_one_batch_dim = true;
                    }
                } else {
                    slog::warn << "Input '" << item.get_any_name()
                               << "' doesn't have batch dimension in layout. -b option will be ignored for this input."
                               << slog::endl;
                }
            }
            info_map[name] = info;
        }

        if (batch_size > 1 && !is_there_at_least_one_batch_dim) {
            throw std::runtime_error("-b option is provided in command line, but there's no inputs with batch(B) "
                                     "dimension in input layout, so batch cannot be set. "
                                     "You may specify layout explicitly using -layout option.");
        }

        // Update scale and mean
        std::map<std::string, std::vector<float>> scale_map = parse_scale_or_mean(scale_string, info_map);
        std::map<std::string, std::vector<float>> mean_map = parse_scale_or_mean(mean_string, info_map);

        for (auto& item : info_map) {
            if (item.second.is_image()) {
                if (scale_map.count(item.first)) {
                    item.second.scale = scale_map.at(item.first);
                }
                if (mean_map.count(item.first)) {
                    item.second.mean = mean_map.at(item.first);
                }
            }
        }

        info_maps.push_back(info_map);
    }

    return info_maps;
}

std::vector<benchmark_app::InputsInfo> get_inputs_info(const std::string& shape_string,
                                                       const std::string& layout_string,
                                                       const size_t batch_size,
                                                       const std::string& tensors_shape_string,
                                                       const std::map<std::string, std::vector<std::string>>& fileNames,
                                                       const std::string& scale_string,
                                                       const std::string& mean_string,
                                                       const std::vector<ov::Output<const ov::Node>>& input_info) {
    bool reshape_required = false;
    return get_inputs_info(shape_string,
                           layout_string,
                           batch_size,
                           tensors_shape_string,
                           fileNames,
                           scale_string,
                           mean_string,
                           input_info,
                           reshape_required);
}

void dump_config(const std::string& filename, const std::map<std::string, ov::AnyMap>& config) {
    nlohmann::json jsonConfig;
    for (const auto& item : config) {
        std::string deviceName = item.first;
        for (const auto& option : item.second) {
            // primary property
            std::stringstream strm;
            option.second.print(strm);
            auto property_string = strm.str();
            jsonConfig[deviceName][option.first] = property_string;
        }
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Can't load config file \"" + filename + "\".");
    }

    ofs << jsonConfig;
}

void load_config(const std::string& filename, std::map<std::string, ov::AnyMap>& config) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Can't load config file \"" + filename + "\".");
    }

    nlohmann::json jsonConfig;
    try {
        ifs >> jsonConfig;
    } catch (const std::exception& e) {
        throw std::runtime_error("Can't parse config file \"" + filename + "\".\n" + e.what());
    }

    for (auto item = jsonConfig.cbegin(), end = jsonConfig.cend(); item != end; ++item) {
        const std::string& deviceName = item.key();
        const auto& itemValue = item.value();
        for (auto option = itemValue.cbegin(), itemValueEnd = itemValue.cend(); option != itemValueEnd; ++option) {
            config[deviceName][option.key()] = option.value().get<std::string>();
        }
    }
}

std::string get_extension(const std::string& name) {
    auto extensionPosition = name.rfind('.', name.size());
    return extensionPosition == std::string::npos ? "" : name.substr(extensionPosition + 1, name.size() - 1);
};

bool is_binary_file(const std::string& filePath) {
    auto extension = get_extension(filePath);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return supported_binary_extensions.find(extension) != supported_binary_extensions.end();
}

bool is_numpy_file(const std::string& filePath) {
    auto extension = get_extension(filePath);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return supported_numpy_extensions.find(extension) != supported_numpy_extensions.end();
}

bool is_image_file(const std::string& filePath) {
    auto extension = get_extension(filePath);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return supported_image_extensions.find(extension) != supported_image_extensions.end();
}

bool contains_binaries(const std::vector<std::string>& filePaths) {
    std::vector<std::string> filtered;
    for (auto& filePath : filePaths) {
        if (is_binary_file(filePath)) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> filter_files_by_extensions(const std::vector<std::string>& filePaths,
                                                    const std::unordered_set<std::string>& extensions) {
    std::vector<std::string> filtered;
    for (auto& filePath : filePaths) {
        auto extension = get_extension(filePath);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extensions.find(extension) != extensions.end()) {
            filtered.push_back(filePath);
        }
    }
    return filtered;
}

std::string parameter_name_to_tensor_name(const std::string& name,
                                          const std::vector<ov::Output<const ov::Node>>& inputs_info,
                                          const std::vector<ov::Output<const ov::Node>>& outputs_info) {
    if (std::any_of(inputs_info.begin(), inputs_info.end(), [name](const ov::Output<const ov::Node>& port) {
            try {
                return port.get_names().count(name) > 0;
            } catch (const ov::Exception&) {
                return false;  // Some ports might have no names - so this is workaround
            }
        })) {
        return name;
    } else if (std::any_of(outputs_info.begin(), outputs_info.end(), [name](const ov::Output<const ov::Node>& port) {
                   try {
                       return port.get_names().count(name) > 0;
                   } catch (const ov::Exception&) {
                       return false;  // Some ports might have no names - so this is workaround
                   }
               })) {
        return name;
    } else {
        for (const auto& port : inputs_info) {
            if (name == port.get_node()->get_friendly_name()) {
                return port.get_any_name();
            }
        }
        for (const auto& port : outputs_info) {
            if (name == port.get_node()->get_input_node_ptr(0)->get_friendly_name()) {
                return port.get_any_name();
            }
        }
    }
    throw std::runtime_error("Provided I/O name \"" + name +
                             "\" is not found neither in tensor names nor in nodes names.");
}
