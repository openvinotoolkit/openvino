// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>

#include <openvino/core/partial_shape.hpp>
#include <openvino/openvino.hpp>

#include "tools_helpers.hpp"


static constexpr char help_message[] = "Optional. Print the usage message.";

static constexpr char model_message[] = "Required. Path to the XML model.";

static constexpr char targetDeviceMessage[] =
        "Required. Specify a target device for which executable network will be compiled.\n"
        "                                             Use \"-d HETERO:<comma-separated_devices_list>\" format to "
        "specify HETERO plugin.\n"
        "                                             Use \"-d MULTI:<comma-separated_devices_list>\" format to "
        "specify MULTI plugin.\n"
        "                                             The application looks for a suitable plugin for the specified "
        "device.";

static constexpr char output_message[] = "Optional. Path to the output file. Default value: \"<model_xml_file>.blob\".";

static constexpr char log_level_message[] = "Optional. Log level for OpenVINO library.";

static constexpr char config_message[] = "Optional. Path to the configuration file.";

static constexpr char inputs_precision_message[] = "Optional. Specifies precision for all input layers of the network.";

static constexpr char outputs_precision_message[] =
        "Optional. Specifies precision for all output layers of the network.";

static constexpr char iop_message[] =
        "Optional. Specifies precision for input and output layers by name.\n"
        "                                             Example: -iop \"input:FP16, output:FP16\".\n"
        "                                             Notice that quotes are required.\n"
        "                                             Overwrites precision from ip and op options for specified "
        "layers.";

static constexpr char inputs_layout_message[] = "Optional. Specifies layout for all input layers of the network.";

static constexpr char outputs_layout_message[] = "Optional. Specifies layout for all output layers of the network.";

static constexpr char iol_message[] =
        "Optional. Specifies layout for input and output layers by name.\n"
        "                                             Example: -iol \"input:NCHW, output:NHWC\".\n"
        "                                             Notice that quotes are required.\n"
        "                                             Overwrites layout from il and ol options for specified layers.";

static constexpr char inputs_model_layout_message[] =
        "Optional. Specifies model layout for all input layers of the network.";

static constexpr char outputs_model_layout_message[] =
        "Optional. Specifies model layout for all output layers of the network.";

static constexpr char ioml_message[] =
        "Optional. Specifies model layout for input and output tensors by name.\n"
        "                                             Example: -ionl \"input:NCHW, output:NHWC\".\n"
        "                                             Notice that quotes are required.\n"
        "                                             Overwrites layout from il and ol options for specified layers.";

static const char shape_message[] =
        " Set shape for model input. For example, \"input1[1,3,224,224],input2[1,4]\" or \"[1,3,224,224]\""
        " in case of one input size. This parameter affect model input shape and can be dynamic."
        " For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]."
        " For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].";

static const char override_model_batch_size[] = "Enforce a model to be compiled for batch size";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "", targetDeviceMessage);
DEFINE_string(o, "", output_message);
DEFINE_string(log_level, "", log_level_message);
DEFINE_string(c, "", config_message);
DEFINE_string(ip, "", inputs_precision_message);
DEFINE_string(op, "", outputs_precision_message);
DEFINE_string(iop, "", iop_message);
DEFINE_string(il, "", inputs_layout_message);
DEFINE_string(ol, "", outputs_layout_message);
DEFINE_string(iol, "", iol_message);
DEFINE_string(iml, "", inputs_model_layout_message);
DEFINE_string(oml, "", outputs_model_layout_message);
DEFINE_string(ioml, "", ioml_message);
DEFINE_string(shape, "", shape_message);
DEFINE_uint32(override_model_batch_size, 1, override_model_batch_size);

namespace {
std::vector<std::string> splitStringList(const std::string& str, char delim) {
    if (str.empty())
        return {};

    std::istringstream istr(str);

    std::vector<std::string> result;
    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }
        result.emplace_back(std::move(elem));
    }

    return result;
}

std::map<std::string, std::string> parseArgMap(std::string argMap) {
    argMap.erase(std::remove_if(argMap.begin(), argMap.end(), ::isspace), argMap.end());

    const auto pairs = splitStringList(argMap, ',');

    std::map<std::string, std::string> parsedMap;
    for (auto&& pair : pairs) {
        const auto lastDelimPos = pair.find_last_of(':');
        auto key = pair.substr(0, lastDelimPos);
        auto value = pair.substr(lastDelimPos + 1);

        if (lastDelimPos == std::string::npos || key.empty() || value.empty()) {
            throw std::invalid_argument("Invalid key/value pair " + pair + ". Expected <layer_name>:<value>");
        }

        parsedMap[std::move(key)] = std::move(value);
    }

    return parsedMap;
}
}  // namespace
using supported_type_t = std::unordered_map<std::string, ov::element::Type>;
ov::element::Type getType(std::string value, const supported_type_t& supported_precisions) {
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);

    const auto precision = supported_precisions.find(value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision");
    }

    return precision->second;
}
ov::element::Type getType(const std::string& value) {
    static const supported_type_t supported_types = {
            {"FP32", ov::element::f32}, {"f32", ov::element::f32},      {"FP16", ov::element::f16},
            {"f16", ov::element::f16},  {"BF16", ov::element::bf16},    {"bf16", ov::element::bf16},
            {"U64", ov::element::u64},  {"u64", ov::element::u64},      {"I64", ov::element::i64},
            {"i64", ov::element::i64},  {"U32", ov::element::u32},      {"u32", ov::element::u32},
            {"I32", ov::element::i32},  {"i32", ov::element::i32},      {"U16", ov::element::u16},
            {"u16", ov::element::u16},  {"I16", ov::element::i16},      {"i16", ov::element::i16},
            {"U8", ov::element::u8},    {"u8", ov::element::u8},        {"I8", ov::element::i8},
            {"i8", ov::element::i8},    {"BOOL", ov::element::boolean}, {"boolean", ov::element::boolean},
    };

    return getType(value, supported_types);
}

bool isFP32(const ov::element::Type& type) {
    return type == ov::element::f32;
}

void configurePrePostProcessing(std::shared_ptr<ov::Model>& model, const std::string& ip, const std::string& op,
                                const std::string& iop, const std::string& il, const std::string& ol,
                                const std::string& iol, const std::string& iml, const std::string& oml,
                                const std::string& ioml) {
    auto preprocessor = ov::preprocess::PrePostProcessor(model);
    const auto inputs = model->inputs();
    const auto outputs = model->outputs();

    if (!ip.empty()) {
        auto type = getType(ip);
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).tensor().set_element_type(type);
        }
    }

    if (!op.empty()) {
        auto type = getType(op);
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).tensor().set_element_type(type);
        }
    }

    if (!iop.empty()) {
        const auto user_precisions_map = parseArgMap(iop);
        for (auto&& item : user_precisions_map) {
            const auto& tensor_name = item.first;
            const auto type = getType(item.second);

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensor_name)) {
                    preprocessor.input(i).tensor().set_element_type(type);
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensor_name)) {
                        preprocessor.output(i).tensor().set_element_type(type);
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensor_name);
        }
    }
    if (!il.empty()) {
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).tensor().set_layout(ov::Layout(il));
        }
    }

    if (!ol.empty()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).tensor().set_layout(ov::Layout(ol));
        }
    }

    if (!iol.empty()) {
        const auto user_precisions_map = parseArgMap(iol);
        for (auto&& item : user_precisions_map) {
            const auto& tensor_name = item.first;

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensor_name)) {
                    preprocessor.input(i).tensor().set_layout(ov::Layout(item.second));
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensor_name)) {
                        preprocessor.output(i).tensor().set_layout(ov::Layout(item.second));
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensor_name);
        }
    }

    if (!iml.empty()) {
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).model().set_layout(ov::Layout(iml));
        }
    }

    if (!oml.empty()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).model().set_layout(ov::Layout(oml));
        }
    }

    if (!ioml.empty()) {
        const auto user_precisions_map = parseArgMap(ioml);
        for (auto&& item : user_precisions_map) {
            const auto& tensor_name = item.first;

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensor_name)) {
                    preprocessor.input(i).model().set_layout(ov::Layout(item.second));
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensor_name)) {
                        preprocessor.output(i).model().set_layout(ov::Layout(item.second));
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensor_name);
        }
    }

    model = preprocessor.build();
}

void printInputAndOutputsInfoShort(const ov::Model& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& param : network.get_parameters()) {
        auto l = param->get_layout();
        std::cout << "    " << param->get_friendly_name() << " : " << param->get_element_type() << " / "
                  << param->get_layout().to_string() << " / " << param->get_partial_shape().to_string() << std::endl;
    }
    std::cout << "Network outputs:" << std::endl;
    for (auto&& result : network.get_results()) {
        std::cout << "    " << result->get_friendly_name() << " : " << result->get_element_type() << " / "
                  << result->get_layout().to_string() << " / " << result->get_output_partial_shape(0).to_string()
                  << std::endl;
    }
}

inline std::string fileNameNoExt(const std::string& filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos)
        return filepath;
    return filepath.substr(0, pos);
}

static void showUsage() {
    std::cout << "compile_tool [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options:                             " << std::endl;
    std::cout << "    -h                                       " << help_message << std::endl;
    std::cout << "    -m                           <value>     " << model_message << std::endl;
    std::cout << "    -d                           <value>     " << targetDeviceMessage << std::endl;
    std::cout << "    -o                           <value>     " << output_message << std::endl;
    std::cout << "    -c                           <value>     " << config_message << std::endl;
    std::cout << "    -ip                          <value>     " << inputs_precision_message << std::endl;
    std::cout << "    -op                          <value>     " << outputs_precision_message << std::endl;
    std::cout << "    -iop                        \"<value>\"    " << iop_message << std::endl;
    std::cout << "    -il                          <value>     " << inputs_layout_message << std::endl;
    std::cout << "    -ol                          <value>     " << outputs_layout_message << std::endl;
    std::cout << "    -iol                        \"<value>\"    " << iol_message << std::endl;
    std::cout << "    -iml                         <value>     " << inputs_model_layout_message << std::endl;
    std::cout << "    -oml                         <value>     " << outputs_model_layout_message << std::endl;
    std::cout << "    -ioml                       \"<value>\"    " << ioml_message << std::endl;
    std::cout << "    -shape                       <value>     " << shape_message << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int* argc, char*** argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::invalid_argument("Path to model xml file is required");
    }

    if (FLAGS_d.empty()) {
        throw std::invalid_argument("Target device name is required");
    }

    if (1 < *argc) {
        std::stringstream message;
        message << "Unknown arguments: ";
        for (auto arg = 1; arg < *argc; arg++) {
            message << (*argv)[arg];
            if (arg < *argc) {
                message << " ";
            }
        }
        throw std::invalid_argument(message.str());
    }

    return true;
}

static std::map<std::string, std::string> parseConfigFile(char comment = '#') {
    std::map<std::string, std::string> config;

    std::ifstream file(FLAGS_c);
    if (file.is_open()) {
        std::string option;
        while (std::getline(file, option)) {
            if (option.empty() || option[0] == comment) {
                continue;
            }
            size_t spacePos = option.find_first_of(" \t\n\r");
            OPENVINO_ASSERT(spacePos != std::string::npos, "Failed to find a space separator in "
                                                           "provided plugin config option: " +
                                                                   option);

            std::string key = option.substr(0, spacePos);

            std::string value{};
            size_t valueStart = option.find_first_not_of(" \t\n\r", spacePos);
            OPENVINO_ASSERT(valueStart != std::string::npos, "An invalid config parameter value detected, "
                                                             "it mustn't be empty: " +
                                                                     option);
            size_t valueEnd = option.find_last_not_of(" \t\n\r");
            value = option.substr(valueStart, valueEnd - valueStart + 1);

            config[key] = std::move(value);
        }
    }
    return config;
}

std::string getFileNameFromPath(const std::string& path,
#if defined(_WIN32)
                                const std::string& sep = "\\") {
#else
                                const std::string& sep = "/") {
#endif
    const auto pos = path.rfind(sep);
    if (std::string::npos == pos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

using TimeDiff = std::chrono::milliseconds;


int main(int argc, char* argv[]) {
    try {
        TimeDiff loadNetworkTimeElapsed{0};

        const auto& version = ov::get_openvino_version();
        std::cout << version.description << " version ......... ";
        std::cout << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH
                  << std::endl;

        std::cout << "Build ........... ";
        std::cout << version.buildNumber << std::endl;
        std::cout << "Parsing command-line arguments" << std::endl;
        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }

        ov::Core core;
        std::cout << "Checking FLAGS_LOG_LEVEL " << FLAGS_log_level << std::endl;

        if (!FLAGS_log_level.empty()) {
            std::cout << "Setting log level " << FLAGS_log_level << std::endl;
            ov::log::Level level;
            std::stringstream{FLAGS_log_level} >> level;
            core.set_property(FLAGS_d, ov::log::level(level));
        }

        std::cout << "Reading model" << std::endl;
        auto model = core.read_model(FLAGS_m);
        auto inputs_info = std::const_pointer_cast<ov::Model>(model)->inputs();
        InputsInfo info_map;

        std::cout << "Performing reshape" << std::endl;
        reshape(std::move(inputs_info), info_map, model, FLAGS_shape, FLAGS_override_model_batch_size, FLAGS_d);

        std::cout << "Configuring model pre & post processing" << std::endl;
        configurePrePostProcessing(model, FLAGS_ip, FLAGS_op, FLAGS_iop, FLAGS_il, FLAGS_ol, FLAGS_iol, FLAGS_iml,
                                   FLAGS_oml, FLAGS_ioml);
        std::cout << "Printing Input and Output Info from model" << std::endl;
        printInputAndOutputsInfoShort(*model);
        auto timeBeforeLoadNetwork = std::chrono::steady_clock::now();
        std::cout << "Parsing configuration file" << std::endl;
        auto configs = parseConfigFile();

        std::cout << "Compiling model" << std::endl;
        auto compiledModel = core.compile_model(model, FLAGS_d, {configs.begin(), configs.end()});
        loadNetworkTimeElapsed =
                std::chrono::duration_cast<TimeDiff>(std::chrono::steady_clock::now() - timeBeforeLoadNetwork);
        std::string outputName = FLAGS_o;
        if (outputName.empty()) {
            outputName = getFileNameFromPath(fileNameNoExt(FLAGS_m)) + ".blob";
        }

        std::ofstream outputFile{outputName, std::ios::out | std::ios::binary};
        if (!outputFile.is_open()) {
            std::cout << "Outputting file " << outputName << " can't be opened for writing" << std::endl;
            return EXIT_FAILURE;
        } else {
            std::cout << "Writing into file - " << outputName << std::endl;
            compiledModel.export_model(outputFile);
        }
        std::cout << "Done. LoadNetwork time elapsed: " << loadNetworkTimeElapsed.count() << " ms" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
