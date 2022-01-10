// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <gflags/gflags.h>

#include "inference_engine.hpp"
#include "openvino/openvino.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>
#include <vpu/utils/string.hpp>

#include "samples/common.hpp"
#include "samples/args_helper.hpp"

static constexpr char help_message[] =
                                             "Optional. Print the usage message.";

static constexpr char model_message[] =
                                             "Required. Path to the XML model.";

static constexpr char targetDeviceMessage[] =
                                             "Required. Specify a target device for which executable network will be compiled.\n"
"                                             Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin.\n"
"                                             Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin.\n"
"                                             The application looks for a suitable plugin for the specified device.";

static constexpr char output_message[] =
                                             "Optional. Path to the output file. Default value: \"<model_xml_file>.blob\".";

static constexpr char log_level_message[] =
                                             "Optional. Log level for InferenceEngine library.";

static constexpr char config_message[] =
                                             "Optional. Path to the configuration file.";

static constexpr char inputs_precision_message[] =
                                             "Optional. Specifies precision for all input layers of the network.";

static constexpr char outputs_precision_message[] =
                                             "Optional. Specifies precision for all output layers of the network.";

static constexpr char iop_message[] =
                                             "Optional. Specifies precision for input and output layers by name.\n"
"                                             Example: -iop \"input:FP16, output:FP16\".\n"
"                                             Notice that quotes are required.\n"
"                                             Overwrites precision from ip and op options for specified layers.";

static constexpr char inputs_layout_message[] =
                                             "Optional. Specifies layout for all input layers of the network.";

static constexpr char outputs_layout_message[] =
                                             "Optional. Specifies layout for all output layers of the network.";

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

static constexpr char api1_message[] =
                                             "Optional. Compile model to legacy format for usage in Inference Engine API,\n"
"                                             by default compiles to OV 2.0 API";

// MYRIAD-specific
static constexpr char number_of_shaves_message[] =
                                             "Optional. Specifies number of shaves.\n"
"                                             Should be set with \"VPU_NUMBER_OF_CMX_SLICES\".\n"
"                                             Overwrites value from config.\n";

static constexpr char number_of_cmx_slices_message[] =
                                             "Optional. Specifies number of CMX slices.\n"
"                                             Should be set with \"VPU_NUMBER_OF_SHAVES\".\n"
"                                             Overwrites value from config.";

static constexpr char tiling_cmx_limit_message[] =
                                             "Optional. Specifies CMX limit for data tiling.\n"
"                                             Value should be equal or greater than -1.\n"
"                                             Overwrites value from config.";

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
DEFINE_bool(ov_api_1_0, false, api1_message);
DEFINE_string(VPU_NUMBER_OF_SHAVES, "", number_of_shaves_message);
DEFINE_string(VPU_NUMBER_OF_CMX_SLICES, "", number_of_cmx_slices_message);
DEFINE_string(VPU_TILING_CMX_LIMIT_KB, "", tiling_cmx_limit_message);

static void showUsage() {
    std::cout << "compile_tool [OPTIONS]" << std::endl;
    std::cout                                                                                      << std::endl;
    std::cout << " Common options:                             "                                   << std::endl;
    std::cout << "    -h                                       "   << help_message                 << std::endl;
    std::cout << "    -m                           <value>     "   << model_message                << std::endl;
    std::cout << "    -d                           <value>     "   << targetDeviceMessage          << std::endl;
    std::cout << "    -o                           <value>     "   << output_message               << std::endl;
    std::cout << "    -c                           <value>     "   << config_message               << std::endl;
    std::cout << "    -ip                          <value>     "   << inputs_precision_message     << std::endl;
    std::cout << "    -op                          <value>     "   << outputs_precision_message    << std::endl;
    std::cout << "    -iop                        \"<value>\"    "   << iop_message                << std::endl;
    std::cout << "    -il                          <value>     "   << inputs_layout_message        << std::endl;
    std::cout << "    -ol                          <value>     "   << outputs_layout_message       << std::endl;
    std::cout << "    -iol                        \"<value>\"    "   << iol_message                << std::endl;
    std::cout << "    -iml                         <value>     "   << inputs_model_layout_message  << std::endl;
    std::cout << "    -oml                         <value>     "   << outputs_model_layout_message << std::endl;
    std::cout << "    -ioml                       \"<value>\"    "   << ioml_message               << std::endl;
    std::cout << "    -ov_api_1_0                              "   << api1_message                 << std::endl;
    std::cout                                                                                      << std::endl;
    std::cout << " MYRIAD-specific options:                    "                                   << std::endl;
    std::cout << "      -VPU_NUMBER_OF_SHAVES      <value>     "   << number_of_shaves_message     << std::endl;
    std::cout << "      -VPU_NUMBER_OF_CMX_SLICES  <value>     "   << number_of_cmx_slices_message << std::endl;
    std::cout << "      -VPU_TILING_CMX_LIMIT_KB   <value>     "   << tiling_cmx_limit_message     << std::endl;
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
            message << argv[arg];
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
            size_t spacePos = option.find(' ');
            std::string key, value;
            if (spacePos != std::string::npos) {
                key = option.substr(0, spacePos);
                value = option.substr(spacePos + 1);
                config[key] = value;
            }
        }
    }
    return config;
}

static std::map<std::string, std::string> configure() {
    const bool isMYRIAD = FLAGS_d.find("MYRIAD") != std::string::npos;

    auto config = parseConfigFile();

    if (isMYRIAD) {
        if (!FLAGS_VPU_NUMBER_OF_SHAVES.empty()) {
            config[InferenceEngine::MYRIAD_NUMBER_OF_SHAVES] = FLAGS_VPU_NUMBER_OF_SHAVES;
        }

        if (!FLAGS_VPU_NUMBER_OF_CMX_SLICES.empty()) {
            config[InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES] = FLAGS_VPU_NUMBER_OF_CMX_SLICES;
        }

        if (!FLAGS_VPU_TILING_CMX_LIMIT_KB.empty()) {
            config[InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB] = FLAGS_VPU_TILING_CMX_LIMIT_KB;
        }
    }

    return config;
}

bool isFP16(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::FP16;
}

bool isFP32(InferenceEngine::Precision precision) {
    return precision == InferenceEngine::Precision::FP32;
}

bool isFloat(InferenceEngine::Precision precision) {
    return isFP16(precision) || isFP32(precision);
}

static void setDefaultIO(InferenceEngine::CNNNetwork& network) {
    const bool isMYRIAD = FLAGS_d.find("MYRIAD") != std::string::npos;
    const bool isVPUX = FLAGS_d.find("VPUX") != std::string::npos;

    if (isMYRIAD) {
        const InferenceEngine::Precision fp16 = InferenceEngine::Precision::FP16;

        for (auto&& layer : network.getInputsInfo()) {
            if (isFloat(layer.second->getPrecision())) {
                layer.second->setPrecision(fp16);
            }
        }

        for (auto&& layer : network.getOutputsInfo()) {
            if (isFloat(layer.second->getPrecision())) {
                layer.second->setPrecision(fp16);
            }
        }
    }

    if (isVPUX) {
        const InferenceEngine::Precision u8 = InferenceEngine::Precision::U8;
        const InferenceEngine::Precision fp32 = InferenceEngine::Precision::FP32;

        for (auto&& layer : network.getInputsInfo()) {
            layer.second->setPrecision(u8);
        }

        for (auto&& layer : network.getOutputsInfo()) {
            layer.second->setPrecision(fp32);
        }
    }
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
    TimeDiff loadNetworkTimeElapsed {0};

    try {
        const auto& version = ov::get_openvino_version();
        std::cout << version.description << " version ......... ";
        std::cout << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH << std::endl;

        std::cout << "Build ........... ";
        std::cout << version.buildNumber << std::endl;

        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }
        if (FLAGS_ov_api_1_0) {
            InferenceEngine::Core ie;
            if (!FLAGS_log_level.empty()) {
                ie.SetConfig({{CONFIG_KEY(LOG_LEVEL), FLAGS_log_level}}, FLAGS_d);
            }

            auto network = ie.ReadNetwork(FLAGS_m);

            setDefaultIO(network);
            processPrecision(network, FLAGS_ip, FLAGS_op, FLAGS_iop);
            processLayout(network, FLAGS_il, FLAGS_ol, FLAGS_iol);

            printInputAndOutputsInfo(network);

            auto timeBeforeLoadNetwork = std::chrono::steady_clock::now();
            auto executableNetwork = ie.LoadNetwork(network, FLAGS_d, configure());
            loadNetworkTimeElapsed = std::chrono::duration_cast<TimeDiff>(std::chrono::steady_clock::now() - timeBeforeLoadNetwork);

            std::string outputName = FLAGS_o;
            if (outputName.empty()) {
                outputName = getFileNameFromPath(fileNameNoExt(FLAGS_m)) + ".blob";
            }

            std::ofstream outputFile{outputName, std::ios::out | std::ios::binary};
            if (!outputFile.is_open()) {
                std::cout << "Output file " << outputName << " can't be opened for writing" << std::endl;
                return EXIT_FAILURE;
            } else {
                executableNetwork.Export(outputFile);
            }
        } else {
            ov::runtime::Core core;
            if (!FLAGS_log_level.empty()) {
                core.set_config({{CONFIG_KEY(LOG_LEVEL), FLAGS_log_level}}, FLAGS_d);
            }

            auto model = core.read_model(FLAGS_m);

            configurePrePostProcessing(model, FLAGS_ip, FLAGS_op, FLAGS_iop, FLAGS_il, FLAGS_ol, FLAGS_iol, FLAGS_iml, FLAGS_oml, FLAGS_ioml);
            printInputAndOutputsInfo(*model);
            auto timeBeforeLoadNetwork = std::chrono::steady_clock::now();
            auto compiledModel = core.compile_model(model, FLAGS_d, configure());
            loadNetworkTimeElapsed = std::chrono::duration_cast<TimeDiff>(std::chrono::steady_clock::now() - timeBeforeLoadNetwork);
            std::string outputName = FLAGS_o;
            if (outputName.empty()) {
                outputName = getFileNameFromPath(fileNameNoExt(FLAGS_m)) + ".blob";
            }

            std::ofstream outputFile{outputName, std::ios::out | std::ios::binary};
            if (!outputFile.is_open()) {
                std::cout << "Output file " << outputName << " can't be opened for writing" << std::endl;
                return EXIT_FAILURE;
            } else {
                compiledModel.export_model(outputFile);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Done. LoadNetwork time elapsed: " << loadNetworkTimeElapsed.count() << " ms" << std::endl;
    return EXIT_SUCCESS;
}
