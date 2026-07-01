// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "inference.hpp"
#include "utils.hpp"

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/op/parameter.hpp"

#include "samples/affinity_utils.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
// clang-format on

namespace {

using Time = std::chrono::steady_clock;

double get_duration_ms(const Time::time_point& start_time) {
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(Time::now() - start_time);
    return static_cast<double>(duration.count()) / 1000.0;
}

bool is_virtual_device_name(const std::string& device_name) {
    static const std::vector<std::string> meta_plugins{"MULTI", "HETERO", "AUTO"};
    return std::find(meta_plugins.begin(), meta_plugins.end(), device_name) != meta_plugins.end();
}

std::vector<std::string> split_string(const std::string& value, char delim) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;

    while (std::getline(stream, item, delim)) {
        result.push_back(item);
    }

    return result;
}

std::string format_any_value(const ov::Any& value);

std::string format_any_map(const ov::AnyMap& values) {
    std::stringstream stream;

    stream << "{";
    for (auto property = values.begin(); property != values.end(); ++property) {
        if (property != values.begin()) {
            stream << ", ";
        }
        stream << property->first << ": " << format_any_value(property->second);
    }
    stream << "}";

    return stream.str();
}

std::string format_any_value(const ov::Any& value) {
    if (value.empty()) {
        return "EMPTY VALUE";
    }

    if (value.is<ov::AnyMap>()) {
        return format_any_map(value.as<ov::AnyMap>());
    }

    std::stringstream stream;
    try {
        value.print(stream);
    } catch (const std::exception& ex) {
        return std::string("<error formatting value: ") + ex.what() + ">";
    }

    auto result = stream.str();
    if (result.empty() && value.is<std::string>()) {
        return "\"\"";
    }
    if (result.empty()) {
        return "<value is not printable as text>";
    }

    return result;
}

std::vector<std::string> parse_device_list(const std::string& device_name) {
    std::string comma_separated_devices = device_name;
    auto colon = comma_separated_devices.find(':');
    std::vector<std::string> devices;
    if (colon != std::string::npos) {
        const auto target_device = comma_separated_devices.substr(0, colon);
        if (is_virtual_device_name(target_device)) {
            devices.push_back(target_device);
        }
        comma_separated_devices = comma_separated_devices.substr(colon + 1);
    }

    for (auto&& device : split_string(comma_separated_devices, ',')) {
        constexpr const char* ws = " \t\n\r\f\v";
        const auto begin = device.find_first_not_of(ws);
        const auto end = device.find_last_not_of(ws);
        device = (begin == std::string::npos) ? std::string{} : device.substr(begin, end - begin + 1);
        if (!device.empty() && device.front() == '-') {
            device.erase(device.begin());
        }
        const auto paren = device.find('(');
        if (paren != std::string::npos) {
            device = device.substr(0, paren);
            const auto trimmed_end = device.find_last_not_of(ws);
            device = trimmed_end == std::string::npos ? std::string{} : device.substr(0, trimmed_end + 1);
        }
        if (!device.empty()) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<std::string> get_hardware_devices(const std::string& device_name) {
    auto devices = parse_device_list(device_name);
    devices.erase(std::remove_if(devices.begin(),
                                 devices.end(),
                                 [](const std::string& device) {
                                     return is_virtual_device_name(device);
                                 }),
                  devices.end());
    return devices;
}

ov::Layout get_default_layout(const ov::PartialShape& shape) {
    if (shape.rank().is_dynamic()) {
        return {};
    }

    if (shape.size() == 3) {
        return shape[2].get_max_length() <= 4 && shape[0].get_max_length() > 4 ? ov::Layout("HWC") : ov::Layout("CHW");
    }

    if (shape.size() == 4) {
        return shape[3].get_max_length() <= 4 && shape[1].get_max_length() > 4 ? ov::Layout("NHWC")
                                                                               : ov::Layout("NCHW");
    }

    return {};
}

size_t get_kv_sequence_length_from_data_shapes(const std::map<std::string, ov::PartialShape>& data_shapes) {
    for (const auto& item : data_shapes) {
        if (!contains_substring(item.first, "past_key") && !contains_substring(item.first, "past_value")) {
            continue;
        }
        if (item.second.is_static() && item.second.size() == 4) {
            return item.second[2].get_length();
        }
    }

    return 0;
}

void prepare_model(std::shared_ptr<ov::Model>& model,
                   const std::string& shape_string,
                   const std::string& data_shape_string) {
    for (auto& input : model->inputs()) {
        if (input.get_tensor().get_names().empty()) {
            input.get_tensor_ptr()->set_names(std::unordered_set<std::string>{input.get_node_shared_ptr()->get_name()});
        }
    }

    if (!shape_string.empty() || !data_shape_string.empty()) {
        std::map<std::string, ov::PartialShape> shapes;
        if (!data_shape_string.empty()) {
            shapes = parse_input_shapes(data_shape_string, std::const_pointer_cast<const ov::Model>(model)->inputs());
            const auto kv_sequence_length = get_kv_sequence_length_from_data_shapes(shapes);
            if (kv_sequence_length != 0) {
                for (const auto& input : model->inputs()) {
                    const auto partial_shape = input.get_partial_shape();
                    if (partial_shape.rank().is_dynamic() || partial_shape.size() != 3) {
                        continue;
                    }

                    const auto name = input.get_any_name();
                    if (contains_substring(name, "embed") || contains_substring(name, "hidden")) {
                        shapes[name] = ov::PartialShape{partial_shape[0],
                                                        static_cast<ov::Dimension::value_type>(kv_sequence_length),
                                                        partial_shape[2]};
                    }
                }
            }
        }
        if (!shape_string.empty()) {
            const auto explicit_shapes =
                parse_input_shapes(shape_string, std::const_pointer_cast<const ov::Model>(model)->inputs());
            for (const auto& item : explicit_shapes) {
                shapes[item.first] = item.second;
            }
        }

        slog::info << "Reshaping model: " << partial_shapes_to_string(shapes) << slog::endl;
        const auto reshape_start_time = Time::now();
        model->reshape(shapes);
        const auto reshape_time_ms = get_duration_ms(reshape_start_time);
        slog::info << "Reshape model took " << format_duration_ms(reshape_time_ms) << " ms" << slog::endl;
    }

    ov::preprocess::PrePostProcessor preproc(model);
    for (const auto& input : model->inputs()) {
        auto layout = dynamic_cast<const ov::op::v0::Parameter&>(*input.get_node()).get_layout();
        if (layout.empty()) {
            layout = get_default_layout(input.get_partial_shape());
            if (!layout.empty()) {
                slog::warn << input.get_any_name() << ": layout is not set explicitly, so it is defaulted to "
                           << layout.to_string() << "." << slog::endl;
            }
        }

        if (!layout.empty()) {
            preproc.input(input.get_any_name()).model().set_layout(layout);
        }
    }

    model = preproc.build();
}

void print_model_info(const ov::Model& model) {
    slog::info << "model name: " << model.get_friendly_name() << slog::endl;

    slog::info << "    inputs" << slog::endl;
    for (const auto& input : model.inputs()) {
        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        slog::info << "        input name: " << name << slog::endl;
        slog::info << "        input type: " << input.get_element_type() << slog::endl;
        slog::info << "        input shape: " << input.get_partial_shape() << slog::endl;
    }

    slog::info << "    outputs" << slog::endl;
    for (const auto& output : model.outputs()) {
        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        slog::info << "        output name: " << name << slog::endl;
        slog::info << "        output type: " << output.get_element_type() << slog::endl;
        slog::info << "        output shape: " << output.get_partial_shape() << slog::endl;
    }
}

void print_affinity_summary(const ov::Model& model) {
    std::map<std::string, size_t> affinities;
    size_t assigned_ops = 0;

    for (const auto& node : model.get_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto affinity = rt_info.find("affinity");
        if (affinity == rt_info.end()) {
            continue;
        }

        affinities[affinity->second.as<std::string>()]++;
        assigned_ops++;
    }

    slog::info << "Affinity assigned to " << assigned_ops << " ops" << slog::endl;
    for (const auto& item : affinities) {
        slog::info << "    " << item.first << " : " << item.second << " ops" << slog::endl;
    }
}

void configure_performance_hint(const std::string& device_name,
                                const std::string& performance_hint,
                                const ov::Core& core,
                                ov::AnyMap& config) {
    if (performance_hint.empty()) {
        return;
    }

    if (performance_hint == "none") {
        config.erase(ov::hint::performance_mode.name());
        return;
    }

    ov::hint::PerformanceMode performance_mode;
    if (performance_hint == "throughput" || performance_hint == "tput") {
        performance_mode = ov::hint::PerformanceMode::THROUGHPUT;
    } else if (performance_hint == "latency") {
        performance_mode = ov::hint::PerformanceMode::LATENCY;
    } else {
        OPENVINO_THROW("Incorrect performance hint. Please set -hint option to throughput(tput), latency, or none.");
    }

    const auto supported_properties = core.get_property(device_name, ov::supported_properties);
    if (std::find(supported_properties.begin(), supported_properties.end(), ov::hint::performance_mode) !=
        supported_properties.end()) {
        config[ov::hint::performance_mode.name()] = performance_mode;
    } else {
        slog::warn << "Device(" << device_name << ") does not support performance hint property(-hint)." << slog::endl;
    }
}

void assign_missing_affinities_to_device(const std::shared_ptr<ov::Model>& model, const std::string& device_name) {
    size_t assigned_ops = 0;

    for (const auto& node : model->get_ops()) {
        auto& rt_info = node->get_rt_info();
        if (rt_info.find("affinity") != rt_info.end()) {
            continue;
        }

        rt_info["affinity"] = device_name;
        assigned_ops++;
    }

    slog::info << "Assigned fallback affinity \"" << device_name << "\" to " << assigned_ops << " unmapped ops"
               << slog::endl;
}

void print_runtime_parameters(const ov::CompiledModel& compiled_model) {
    slog::info << "Model:" << slog::endl;

    const auto supported_properties = compiled_model.get_property(ov::supported_properties);
    for (const auto& property_name : supported_properties) {
        if (property_name == ov::supported_properties) {
            continue;
        }

        const auto property = compiled_model.get_property(property_name);
        if (property_name == ov::device::properties && property.is<ov::AnyMap>()) {
            const auto devices_properties = property.as<ov::AnyMap>();
            for (const auto& device_properties : devices_properties) {
                slog::info << "  " << device_properties.first << ":" << slog::endl;
                if (device_properties.second.is<ov::AnyMap>()) {
                    for (const auto& device_property : device_properties.second.as<ov::AnyMap>()) {
                        slog::info << "    " << device_property.first << ": "
                                   << format_any_value(device_property.second) << slog::endl;
                    }
                } else {
                    slog::info << "    " << format_any_value(device_properties.second) << slog::endl;
                }
            }
        } else {
            slog::info << "  " << property_name << ": " << format_any_value(property) << slog::endl;
        }
    }
}

size_t parse_iterations(const std::string& value) {
    if (value.empty() || value.front() == '-') {
        OPENVINO_THROW("--niter requires a positive integer value.");
    }

    try {
        size_t parsed_characters = 0;
        const auto iterations = std::stoull(value, &parsed_characters);
        if (parsed_characters != value.size() || iterations == 0 || iterations > std::numeric_limits<size_t>::max()) {
            OPENVINO_THROW("--niter requires a positive integer value.");
        }

        return static_cast<size_t>(iterations);
    } catch (const std::exception&) {
        OPENVINO_THROW("--niter requires a positive integer value.");
    }
}

void print_usage(const std::string& executable_name) {
    slog::info << "Usage : " << executable_name << " -m <path_to_model> [-d <device_name>] "
               << "[-affinity <affinity|path_to_affinity_json>] [--fallback-device <device>] "
               << "[-hint <performance_hint>] [-shape <shapes>] [-data_shape <shapes>|--data-shape <shapes>] "
               << "[-niter <integer>] [-no_warmup]" << slog::endl;
}

std::string get_option_value(int argc, tchar* argv[], int& arg_index, const std::string& option_name) {
    if (++arg_index == argc) {
        throw std::logic_error(option_name + " requires a value.");
    }

    return TSTRING2STRING(argv[arg_index]);
}

}  // namespace

int tmain(int argc, tchar* argv[]) {
    try {
        const std::function<void(std::string_view)> log_callback{[](std::string_view msg) {
            slog::info << msg;
        }};
        ov::util::set_log_callback(log_callback);

        slog::info << ov::get_openvino_version() << slog::endl;

        const std::string executable_name = TSTRING2STRING(argv[0]);
        if (argc == 1) {
            print_usage(executable_name);
            return EXIT_FAILURE;
        }

        std::vector<std::string> positional_arguments;
        std::string model_path;
        std::string device_name = "CPU";
        std::string affinity_spec;
        std::string fallback_device;
        std::string performance_hint;
        std::string shape_string;
        std::string data_shape_string;
        size_t iterations = 1;
        bool skip_warmup = false;

        for (int arg_index = 1; arg_index < argc; ++arg_index) {
            const std::string option = TSTRING2STRING(argv[arg_index]);
            if (option == "-h" || option == "--help") {
                print_usage(executable_name);
                return EXIT_SUCCESS;
            } else if (option == "-m" || option == "--model") {
                model_path = get_option_value(argc, argv, arg_index, option);
            } else if (option == "-d" || option == "--device") {
                device_name = get_option_value(argc, argv, arg_index, option);
            } else if (option == "-affinity" || option == "--affinity") {
                affinity_spec = get_option_value(argc, argv, arg_index, option);
            } else if (option == "--fallback-device") {
                fallback_device = get_option_value(argc, argv, arg_index, option);
            } else if (option == "-hint" || option == "--hint") {
                performance_hint = to_lower(get_option_value(argc, argv, arg_index, option));
            } else if (option == "-shape" || option == "--shape") {
                shape_string = get_option_value(argc, argv, arg_index, option);
            } else if (option == "-data_shape" || option == "--data_shape" || option == "--data-shape") {
                data_shape_string = get_option_value(argc, argv, arg_index, option);
            } else if (option == "-niter" || option == "--niter") {
                iterations = parse_iterations(get_option_value(argc, argv, arg_index, option));
            } else if (option == "-no_warmup" || option == "--no_warmup") {
                skip_warmup = true;
            } else if (!option.empty() && option.front() != '-') {
                positional_arguments.push_back(option);
            } else {
                throw std::logic_error("Unsupported option: " + option +
                                       ". Expected -m, -d, -affinity, --fallback-device, -hint, -shape, "
                                       "-data_shape, -niter, or -no_warmup.");
            }
        }

        if (model_path.empty() && !positional_arguments.empty()) {
            model_path = positional_arguments.front();
        }
        if (positional_arguments.size() > 1) {
            device_name = positional_arguments[1];
        }
        if (affinity_spec.empty() && positional_arguments.size() > 2) {
            affinity_spec = positional_arguments[2];
        }
        if (positional_arguments.size() > 3) {
            throw std::logic_error("Too many positional arguments. Use -m, -d, and -affinity options instead.");
        }
        if (model_path.empty()) {
            print_usage(executable_name);
            throw std::logic_error("Model path is required. Please provide it with -m <path_to_model>.");
        }
        if (device_name.empty()) {
            throw std::logic_error("Device name must not be empty.");
        }
        const auto parsed_devices = parse_device_list(device_name);
        const auto hardware_devices = get_hardware_devices(device_name);
        if (parsed_devices.empty()) {
            throw std::logic_error("Failed to parse device name: " + device_name);
        }
        if (!fallback_device.empty() && affinity_spec.empty()) {
            throw std::logic_error("The --fallback-device option requires -affinity. "
                                   "Please provide an affinity JSON file or remove --fallback-device.");
        }
        if (!affinity_spec.empty() && (parsed_devices.front() != "HETERO" || hardware_devices.empty())) {
            throw std::logic_error(
                "The -affinity option is supported only with the HETERO plugin and requires an explicit device list. "
                "Please use -d HETERO:<devices> or remove -affinity.");
        }
        if (!fallback_device.empty() &&
            std::find(hardware_devices.begin(), hardware_devices.end(), fallback_device) == hardware_devices.end()) {
            throw std::logic_error("--fallback-device must be one of the devices listed in -d HETERO:<devices>.");
        }

        ov::Core core;

        slog::info << "Loading model files" << slog::endl;
        const auto read_start_time = Time::now();
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        const auto read_time_ms = get_duration_ms(read_start_time);
        slog::info << "Read model took " << format_duration_ms(read_time_ms) << " ms" << slog::endl;
        print_model_info(*model);
        prepare_model(model, shape_string, data_shape_string);
        print_model_info(*model);

        if (!affinity_spec.empty()) {
            apply_manual_affinities(model, affinity_spec, hardware_devices, fallback_device.empty());
            print_affinity_summary(*model);
        } else {
            slog::info << "No manual affinity specified" << slog::endl;
        }

        ov::AnyMap device_config;
        configure_performance_hint(parsed_devices.front(), performance_hint, core, device_config);

        if (!fallback_device.empty()) {
            assign_missing_affinities_to_device(model, fallback_device);
            print_affinity_summary(*model);
        }

        const auto compile_start_time = Time::now();
        ov::CompiledModel compiled_model = core.compile_model(model, device_name, device_config);
        const auto compile_time_ms = get_duration_ms(compile_start_time);

        slog::info << "Model compiled successfully" << slog::endl;
        slog::info << "Compile model took " << compile_time_ms << " ms" << slog::endl;
        print_runtime_parameters(compiled_model);

        for (const auto& output : compiled_model.outputs()) {
            slog::info << output.get_any_name() << " : " << output.get_element_type() << " / "
                       << output.get_partial_shape() << slog::endl;
        }

        run_inference(compiled_model, data_shape_string, iterations, skip_warmup);
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
