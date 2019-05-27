//
// Copyright (C) 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <gflags/gflags.h>

#include "inference_engine.hpp"
#include <vpu/private_plugin_config.hpp>
#include "samples/common.hpp"
#include "vpu/utils/string.hpp"

#include "vpu_tools_common.hpp"

static constexpr char help_message[] = "Optional. Print a usage message.";
static constexpr char model_message[] = "Required. Path to xml model.";
static constexpr char plugin_path_message[] = "Optional. Path to a plugin folder.";
static constexpr char output_message[] = "Optional. Path to the output file. Default value: \"<model_xml_file>.blob\".";
static constexpr char config_message[] = "Optional. Path to the configuration file. Default value: \"config\".";
static constexpr char platform_message[] = "Optional. Specifies movidius platform."
                                           " Supported values: VPU_2450, VPU_2480."
                                           " Overwrites value from config.";
static constexpr char number_of_shaves_message[] = "Optional. Specifies number of shaves."
                                                   " Should be set with \"VPU_NUMBER_OF_CMX_SLICES\"."
                                                   " Overwrites value from config.";
static constexpr char number_of_cmx_slices_message[] = "Optional. Specifies number of CMX slices."
                                                       " Should be set with \"VPU_NUMBER_OF_SHAVES\"."
                                                       " Overwrites value from config.";
static constexpr char inputs_precision_message[] = "Optional. Specifies precision for all input layers of network."
                                                   " Supported values: FP32, FP16, U8. Default value: FP16.";
static constexpr char outputs_precision_message[] = "Optional. Specifies precision for all output layers of network."
                                                    " Supported values: FP32, FP16, U8. Default value: FP16.";
static constexpr char iop_message[] = "Optional. Specifies precision for input/output layers by name.\n"
"                                             By default all inputs and outputs have FP16 precision.\n"
"                                             Available precisions: FP32, FP16, U8.\n"
"                                             Example: -iop \"input:FP16, output:FP16\".\n"
"                                             Notice that quotes are required.\n"
"                                             Overwrites precision from ip and op options for specified layers.";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", help_message);
DEFINE_string(pp, "", plugin_path_message);
DEFINE_string(o, "", output_message);
DEFINE_string(c, "config", config_message);
DEFINE_string(ip, "", inputs_precision_message);
DEFINE_string(op, "", outputs_precision_message);
DEFINE_string(iop, "", iop_message);
DEFINE_string(VPU_PLATFORM, "", platform_message);
DEFINE_string(VPU_NUMBER_OF_SHAVES, "", number_of_shaves_message);
DEFINE_string(VPU_NUMBER_OF_CMX_SLICES, "", number_of_cmx_slices_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "myriad_compile [OPTIONS]" << std::endl;
    std::cout << "[OPTIONS]:" << std::endl;
    std::cout << "    -h                                       " << help_message << std::endl;
    std::cout << "    -m                           <value>     " << model_message << std::endl;
    std::cout << "    -pp                          <value>     " << plugin_path_message << std::endl;
    std::cout << "    -o                           <value>     " << output_message << std::endl;
    std::cout << "    -c                           <value>     " << config_message << std::endl;
    std::cout << "    -ip                          <value>     " << inputs_precision_message << std::endl;
    std::cout << "    -op                          <value>     " << outputs_precision_message << std::endl;
    std::cout << "    -iop                        \"<value>\"    " << iop_message << std::endl;
    std::cout << "    -VPU_PLATFORM                <value>     " << platform_message << std::endl;
    std::cout << "    -VPU_NUMBER_OF_SHAVES        <value>     " << number_of_shaves_message << std::endl;
    std::cout << "    -VPU_NUMBER_OF_CMX_SLICES    <value>     " << number_of_cmx_slices_message << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int *argc, char ***argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::invalid_argument("Path to model xml file is required");
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

static std::map<std::string, std::string> configure(const std::string &configFile, const std::string &xmlFileName) {
    auto config = parseConfig(configFile);

    if (!FLAGS_VPU_PLATFORM.empty()) {
        config[VPU_CONFIG_KEY(PLATFORM)] = FLAGS_VPU_PLATFORM;
    }

    if (!FLAGS_VPU_NUMBER_OF_SHAVES.empty()) {
        config[VPU_CONFIG_KEY(NUMBER_OF_SHAVES)] = FLAGS_VPU_NUMBER_OF_SHAVES;
    }

    if (!FLAGS_VPU_NUMBER_OF_CMX_SLICES.empty()) {
        config[VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES)] = FLAGS_VPU_NUMBER_OF_CMX_SLICES;
    }

    auto modelConfigFile = fileNameNoExt(xmlFileName) + ".conf.xml";
    {
        std::ifstream file(modelConfigFile);
        if (!file.is_open()) {
            modelConfigFile.clear();
        }
    }

    if (!modelConfigFile.empty()) {
        config[VPU_CONFIG_KEY(NETWORK_CONFIG)] = "file=" + modelConfigFile;
    }

    return config;
}

static std::map<std::string, std::string> parsePrecisions(const std::string &iop) {
    std::string user_input = iop;
    user_input.erase(std::remove_if(user_input.begin(), user_input.end(), ::isspace), user_input.end());

    std::vector<std::string> inputs;
    vpu::splitStringList(user_input, inputs, ',');

    std::map<std::string, std::string> precisions;
    for (auto &&input : inputs) {
        std::vector<std::string> precision;
        vpu::splitStringList(input, precision, ':');
        if (precision.size() != 2) {
            throw std::invalid_argument("Invalid precision " + input + ". Expected layer_name : precision_value");
        }

        precisions[precision[0]] = precision[1];
    }

    return precisions;
}

static InferenceEngine::Precision getPrecision(const std::string &value) {
    static const std::unordered_map<std::string, InferenceEngine::Precision> supported_precisions = {
         { "FP32", InferenceEngine::Precision::FP32 },
         { "FP16", InferenceEngine::Precision::FP16 },
         { "U8", InferenceEngine::Precision::U8 }
    };

    std::string upper_value = value;
    std::transform(value.begin(), value.end(), upper_value.begin(), ::toupper);
    auto precision = supported_precisions.find(upper_value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error(value + " is not a valid precision");
    }

    return precision->second;
}

void setPrecisions(const InferenceEngine::CNNNetwork &network, const std::string &iop) {
    auto precisions = parsePrecisions(iop);
    auto inputs = network.getInputsInfo();
    auto outputs = network.getOutputsInfo();

    for (auto &&layer : precisions) {
        auto name = layer.first;
        auto precision = getPrecision(layer.second);

        auto input_precision = inputs.find(name);
        auto output_precision = outputs.find(name);

        if (input_precision != inputs.end()) {
            input_precision->second->setPrecision(precision);
        } else if (output_precision != outputs.end()) {
            output_precision->second->setPrecision(precision);
        } else {
            throw std::logic_error(name + " is not an input neither output");
        }
    }
}

static void processPrecisions(InferenceEngine::CNNNetwork &network,
                              const std::string &inputs_precision, const std::string &outputs_precision,
                              const std::string &iop) {
    setPrecisions(network);

    if (!inputs_precision.empty()) {
        auto precision = getPrecision(inputs_precision);
        for (auto &&layer : network.getInputsInfo()) {
            layer.second->setPrecision(precision);
        }
    }

    if (!outputs_precision.empty()) {
        auto precision = getPrecision(outputs_precision);
        for (auto &&layer : network.getOutputsInfo()) {
            layer.second->setPrecision(precision);
        }
    }

    if (!iop.empty()) {
        setPrecisions(network, iop);
    }
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "Inference Engine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }

        auto network = readNetwork(FLAGS_m);

        processPrecisions(network, FLAGS_ip, FLAGS_op, FLAGS_iop);

        auto plugin = loadPlugin("MYRIAD", FLAGS_pp);
        InferenceEngine::ExecutableNetwork executableNetwork = plugin.LoadNetwork(network, configure(FLAGS_c, FLAGS_m));

        std::string outputName = FLAGS_o;
        if (outputName.empty()) {
            outputName = fileNameNoExt(FLAGS_m) + ".blob";
        }
        executableNetwork.Export(outputName);
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
