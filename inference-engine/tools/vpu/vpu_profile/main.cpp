/*
// Copyright (c) 2018-2019 Intel Corporation
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
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <map>
#include <cmath>
#include <future>
#include <atomic>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

#include <gflags/gflags.h>

#include "inference_engine.hpp"
#include "precision_utils.h"

#include "vpu_tools_common.hpp"
#include "vpu/vpu_plugin_config.hpp"
#include "vpu/private_plugin_config.hpp"
#include "samples/common.hpp"

static constexpr char help_message[]        = "Print a help(this) message.";
static constexpr char model_message[]       = "Path to xml model.";
static constexpr char inputs_dir_message[]  = "Path to folder with images, only bitmap(.bmp) supported. Default: \".\".";
static constexpr char config_message[]      = "Path to the configuration file. Default value: \"config\".";
static constexpr char iterations_message[]  = "Specifies number of iterations. Default value: 16.";
static constexpr char plugin_message[]      = "Specifies plugin. Supported values: myriad, hddl.\n"
    "\t            \t         \tDefault value: \"myriad\".";
static constexpr char report_message[]      = "Specifies report type. Supported values: per_layer, per_stage.\n"
    "\t            \t         \tOverrides value in configuration file if provided. Default value: \"per_stage\"";

DEFINE_bool(h,                false, help_message);
DEFINE_string(model,             "", model_message);
DEFINE_string(inputs_dir,       ".", inputs_dir_message);
DEFINE_string(config,            "", config_message);
DEFINE_int32(iterations,         16, iterations_message);
DEFINE_string(plugin,      "myriad", plugin_message);
DEFINE_string(report,      "", report_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "vpu_profile [OPTIONS]" << std::endl;
    std::cout << "[OPTIONS]:" << std::endl;
    std::cout << "\t-h          \t         \t"   << help_message        << std::endl;
    std::cout << "\t-model      \t <value> \t"   << model_message       << std::endl;
    std::cout << "\t-inputs_dir \t <value> \t"   << inputs_dir_message  << std::endl;
    std::cout << "\t-config     \t <value> \t"   << config_message      << std::endl;
    std::cout << "\t-iterations \t <value> \t"   << iterations_message  << std::endl;
    std::cout << "\t-plugin     \t <value> \t"   << plugin_message      << std::endl;
    std::cout << "\t-report     \t <value> \t"   << report_message      << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int *argc, char ***argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_model.empty()) {
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

static std::map<std::string, std::string> configure(const std::string& confFileName, const std::string& report) {
    auto config = parseConfig(confFileName);

    /* Since user can specify config file we probably can avoid it */
    config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
    config[VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME)] = CONFIG_VALUE(YES);
    /* 
        Default is PER_LAYER
    */
    if (report == "per_layer") {
        config[VPU_CONFIG_KEY(PERF_REPORT_MODE)] = VPU_CONFIG_VALUE(PER_LAYER);
    } else if (report == "per_stage") {
        config[VPU_CONFIG_KEY(PERF_REPORT_MODE)] = VPU_CONFIG_VALUE(PER_STAGE);
    } else if (config.find(VPU_CONFIG_KEY(PERF_REPORT_MODE)) == config.end()) {
        config[VPU_CONFIG_KEY(PERF_REPORT_MODE)] = VPU_CONFIG_VALUE(PER_LAYER);
    }
    return config;
}

template<typename T>
static bool isImage(const T& blob) {
    auto descriptor = blob->getTensorDesc();
    if (descriptor.getLayout() != InferenceEngine::NCHW) {
        return false;
    }

    auto channels = descriptor.getDims()[1];
    return channels == 3;
}

static void loadInputs(std::size_t requestIdx, const std::vector<std::string>& images,
                       const std::vector<std::string>& binaries, InferenceEngine::InferRequest& request,
                       InferenceEngine::CNNNetwork& network) {
    for (auto &&input : network.getInputsInfo()) {
        auto blob = request.GetBlob(input.first);

        if (isImage(blob)) {
            loadImage(images[requestIdx % images.size()], blob);
        } else {
            loadBinaryTensor(binaries[requestIdx % binaries.size()], blob);
        }
    }
}

static std::string process_user_input(const std::string &src) {
    std::string name = src;
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    name.erase(std::remove_if(name.begin(), name.end(), ::isspace), name.end());

    return name;
}

static std::size_t getNumberRequests(const std::string &plugin) {
    static const std::unordered_map<std::string, std::size_t> supported_plugins = {
        { "MYRIAD", 4 }
    };

    auto num_requests = supported_plugins.find(plugin);
    if (num_requests == supported_plugins.end()) {
        throw std::invalid_argument("Unknown plugin " + plugin);
    }

    return num_requests->second;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "Inference Engine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }

        auto network = readNetwork(FLAGS_model);
        setPrecisions(network);

        auto user_plugin = process_user_input(FLAGS_plugin);

        InferenceEngine::Core ie;
        auto executableNetwork = ie.LoadNetwork(network, user_plugin, configure(FLAGS_config, FLAGS_report));

        auto num_requests = getNumberRequests(user_plugin);

        auto images = extractFilesByExtension(FLAGS_inputs_dir, "bmp", 1);
        auto hasImageInput = [](const InferenceEngine::CNNNetwork &network) {
            auto inputs = network.getInputsInfo();
            auto isImageInput = [](const InferenceEngine::InputsDataMap::value_type &input) {
                return isImage(input.second);
            };
            return std::any_of(inputs.begin(), inputs.end(), isImageInput);
        };

        if (hasImageInput(network) && images.empty()) {
            throw std::invalid_argument(FLAGS_inputs_dir + " does not contain images for network");
        }

        auto binaries = extractFilesByExtension(FLAGS_inputs_dir, "bin", 1);
        auto hasBinaryInput = [](const InferenceEngine::CNNNetwork &network) {
            auto inputs = network.getInputsInfo();
            auto isBinaryInput = [](const InferenceEngine::InputsDataMap::value_type &input) {
                return !isImage(input.second);
            };
            return std::any_of(inputs.begin(), inputs.end(), isBinaryInput);
        };

        if (hasBinaryInput(network) && binaries.empty()) {
            throw std::invalid_argument(FLAGS_inputs_dir + " does not contain binaries for network");
        }

        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> performance;

        std::atomic<std::size_t> iteration{0};
        std::promise<void> done;
        bool needStartAsync{true};
        std::size_t profiledIteration = 2 * num_requests + FLAGS_iterations;

        std::vector<InferenceEngine::InferRequest> requests(num_requests);
        std::vector<std::size_t> current_iterations(num_requests);

        using callback_t = std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>;

        for (std::size_t request = 0; request < num_requests; ++request) {
            requests[request] = executableNetwork.CreateInferRequest();
            current_iterations[request] = 0;

            loadInputs(request, images, binaries, requests[request], network);

            callback_t callback =
                [request, profiledIteration, &done, &needStartAsync, &performance, &iteration, &current_iterations]
                (InferenceEngine::InferRequest inferRequest, InferenceEngine::StatusCode code) {
                if (code != InferenceEngine::StatusCode::OK) {
                    THROW_IE_EXCEPTION << "Infer request failed with code " << code;
                }

                auto current_iteration = current_iterations[request];
                if (current_iteration == profiledIteration) {
                    performance = inferRequest.GetPerformanceCounts();
                    needStartAsync = false;
                    done.set_value();
                }

                if (needStartAsync) {
                    current_iterations[request] = iteration++;
                    inferRequest.StartAsync();
                }
            };

            requests[request].SetCompletionCallback<callback_t>(callback);
        }

        auto doneFuture = done.get_future();

        for (std::size_t request = 0; request < num_requests; ++request) {
            current_iterations[request] = iteration++;
            requests[request].StartAsync();
        }

        doneFuture.wait();
        printPerformanceCounts(performance, FLAGS_report);
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
