// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

#include "gna/gna_config.hpp"
#include "gpu/gpu_config.hpp"
#include "vpu/vpu_plugin_config.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "inputs_filling.hpp"
#include "progress_bar.hpp"
#include "remote_tensors_filling.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"
// clang-format on

static const size_t progressBarDefaultTotalCount = 1000;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validating input
    // arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        show_usage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        show_usage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_latency_percentile > 100 || FLAGS_latency_percentile < 1) {
        show_usage();
        throw std::logic_error("The percentile value is incorrect. The applicable values range is [1, 100].");
    }
    if (FLAGS_api != "async" && FLAGS_api != "sync") {
        throw std::logic_error("Incorrect API. Please set -api option to `sync` or `async` value.");
    }
    if (!FLAGS_hint.empty() && FLAGS_hint != "throughput" && FLAGS_hint != "tput" && FLAGS_hint != "latency") {
        throw std::logic_error("Incorrect performance hint. Please set -hint option to"
                               "either `throughput`(tput) or `latency' value.");
    }
    if (!FLAGS_report_type.empty() && FLAGS_report_type != noCntReport && FLAGS_report_type != averageCntReport &&
        FLAGS_report_type != detailedCntReport) {
        std::string err = "only " + std::string(noCntReport) + "/" + std::string(averageCntReport) + "/" +
                          std::string(detailedCntReport) +
                          " report types are supported (invalid -report_type option value)";
        throw std::logic_error(err);
    }

    if ((FLAGS_report_type == averageCntReport) && ((FLAGS_d.find("MULTI") != std::string::npos))) {
        throw std::logic_error("only " + std::string(detailedCntReport) + " report type is supported for MULTI device");
    }

    bool isNetworkCompiled = fileExt(FLAGS_m) == "blob";
    bool isPrecisionSet = !(FLAGS_ip.empty() && FLAGS_op.empty() && FLAGS_iop.empty());
    if (isNetworkCompiled && isPrecisionSet) {
        std::string err = std::string("Cannot set precision for a compiled network. ") +
                          std::string("Please re-compile your network with required precision "
                                      "using compile_tool");

        throw std::logic_error(err);
    }
    return true;
}

static void next_step(const std::string additional_info = "") {
    static size_t step_id = 0;
    static const std::map<size_t, std::string> step_names = {
        {1, "Parsing and validating input arguments"},
        {2, "Loading Inference Engine"},
        {3, "Setting device configuration"},
        {4, "Reading network files"},
        {5, "Resizing network to match image sizes and given batch"},
        {6, "Configuring input of the model"},
        {7, "Loading the model to the device"},
        {8, "Setting optimal runtime parameters"},
        {9, "Creating infer requests and preparing input blobs with data"},
        {10, "Measuring performance"},
        {11, "Dumping statistics report"}};

    step_id++;
    if (step_names.count(step_id) == 0)
        IE_THROW() << "Step ID " << step_id << " is out of total steps number " << step_names.size();

    std::cout << "[Step " << step_id << "/" << step_names.size() << "] " << step_names.at(step_id)
              << (additional_info.empty() ? "" : " (" + additional_info + ")") << std::endl;
}

/**
 * @brief The entry point of the benchmark application
 */
int main(int argc, char* argv[]) {
    std::shared_ptr<StatisticsReport> statistics;
    try {
        ov::runtime::CompiledModel compiledModel;

        // ----------------- 1. Parsing and validating input arguments
        // -------------------------------------------------
        next_step();

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        bool isNetworkCompiled = fileExt(FLAGS_m) == "blob";
        if (isNetworkCompiled) {
            slog::info << "Network is compiled" << slog::endl;
        }

        std::vector<gflags::CommandLineFlagInfo> flags;
        StatisticsReport::Parameters command_line_arguments;
        gflags::GetAllFlags(&flags);
        for (auto& flag : flags) {
            if (!flag.is_default) {
                command_line_arguments.push_back({flag.name, flag.current_value});
            }
        }
        if (!FLAGS_report_type.empty()) {
            statistics =
                std::make_shared<StatisticsReport>(StatisticsReport::Config{FLAGS_report_type, FLAGS_report_folder});
            statistics->add_parameters(StatisticsReport::Category::COMMAND_LINE_PARAMETERS, command_line_arguments);
        }
        auto isFlagSetInCommandLine = [&command_line_arguments](const std::string& name) {
            return (std::find_if(command_line_arguments.begin(),
                                 command_line_arguments.end(),
                                 [name](const std::pair<std::string, std::string>& p) {
                                     return p.first == name;
                                 }) != command_line_arguments.end());
        };

        std::string device_name = FLAGS_d;

        // Parse devices
        auto devices = parse_devices(device_name);

        // Parse nstreams per device
        std::map<std::string, std::string> device_nstreams = parse_nstreams_value_per_device(devices, FLAGS_nstreams);

        // Load device config file if specified
        std::map<std::string, std::map<std::string, std::string>> config;

        if (!FLAGS_load_config.empty()) {
            load_config(FLAGS_load_config, config);
        }

        /** This vector stores paths to the processed images with input names**/
        auto inputFiles = parse_input_arguments(gflags::GetArgvs());

        // ----------------- 2. Loading the Inference Engine
        // -----------------------------------------------------------
        next_step();

        ov::runtime::Core core;

        if (FLAGS_d.find("CPU") != std::string::npos && !FLAGS_l.empty()) {
            // CPU (MKLDNN) extensions is loaded as a shared library
            core.add_extension(FLAGS_l);
            slog::info << "CPU (MKLDNN) extensions is loaded " << FLAGS_l << slog::endl;
        }

        // Load clDNN Extensions
        if ((FLAGS_d.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
            // Override config if command line parameter is specified
            if (!config.count("GPU"))
                config["GPU"] = {};
            config["GPU"][CONFIG_KEY(CONFIG_FILE)] = FLAGS_c;
        }
        if (config.count("GPU") && config.at("GPU").count(CONFIG_KEY(CONFIG_FILE))) {
            auto ext = config.at("GPU").at(CONFIG_KEY(CONFIG_FILE));
            core.set_config({{CONFIG_KEY(CONFIG_FILE), ext}}, "GPU");
            slog::info << "GPU extensions is loaded " << ext << slog::endl;
        }

        if (FLAGS_hint.empty()) {
            for (auto& device : devices) {
                std::vector<std::string> supported_config_keys =
                    core.get_metric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                if (std::find(supported_config_keys.begin(),
                              supported_config_keys.end(),
                              CONFIG_KEY(PERFORMANCE_HINT)) != supported_config_keys.end()) {
                    slog::warn << "-hint default value is determined as " << CONFIG_VALUE(THROUGHPUT)
                               << " automatically for " << device
                               << " device. For more detailed information look at README." << slog::endl;
                    FLAGS_hint = CONFIG_VALUE(THROUGHPUT);
                }
            }
        }

        slog::info << "OpenVINO: " << ov::get_openvino_version() << slog::endl;
        slog::info << "Device info: " << slog::endl;
        slog::info << core.get_versions(device_name) << slog::endl;

        // ----------------- 3. Setting device configuration
        // -----------------------------------------------------------
        next_step();
        std::string ov_perf_hint;
        if (FLAGS_hint == "throughput" || FLAGS_hint == "tput")
            ov_perf_hint = CONFIG_VALUE(THROUGHPUT);
        else if (FLAGS_hint == "latency")
            ov_perf_hint = CONFIG_VALUE(LATENCY);

        auto getDeviceTypeFromName = [](std::string device) -> std::string {
            return device.substr(0, device.find_first_of(".("));
        };

        // Set default values from dumped config
        std::set<std::string> default_devices;
        for (auto& device : devices) {
            auto default_config = config.find(getDeviceTypeFromName(device));
            if (default_config != config.end()) {
                if (!config.count(device)) {
                    config[device] = default_config->second;
                    default_devices.emplace(default_config->first);
                }
            }
        }
        for (auto& device : default_devices) {
            config.erase(device);
        }

        bool perf_counts = false;
        // Update config per device according to command line parameters
        for (auto& device : devices) {
            if (!config.count(device))
                config[device] = {};
            std::map<std::string, std::string>& device_config = config.at(device);

            // high-level performance modes
            if (!ov_perf_hint.empty()) {
                device_config[CONFIG_KEY(PERFORMANCE_HINT)] = ov_perf_hint;
                if (FLAGS_nireq != 0)
                    device_config[CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)] = std::to_string(FLAGS_nireq);
            }

            // Set performance counter
            if (isFlagSetInCommandLine("pc")) {
                // set to user defined value
                device_config[CONFIG_KEY(PERF_COUNT)] = FLAGS_pc ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
            } else if (device_config.count(CONFIG_KEY(PERF_COUNT)) &&
                       (device_config.at(CONFIG_KEY(PERF_COUNT)) == "YES")) {
                slog::warn << "Performance counters for " << device
                           << " device is turned on. To print results use -pc option." << slog::endl;
            } else if (FLAGS_report_type == detailedCntReport || FLAGS_report_type == averageCntReport) {
                slog::warn << "Turn on performance counters for " << device << " device since report type is "
                           << FLAGS_report_type << "." << slog::endl;
                device_config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
            } else if (!FLAGS_exec_graph_path.empty()) {
                slog::warn << "Turn on performance counters for " << device << " device due to execution graph dumping."
                           << slog::endl;
                device_config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
            } else {
                // set to default value
                device_config[CONFIG_KEY(PERF_COUNT)] = FLAGS_pc ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
            }
            perf_counts = (device_config.at(CONFIG_KEY(PERF_COUNT)) == CONFIG_VALUE(YES)) ? true : perf_counts;

            // the rest are individual per-device settings (overriding the values set with perf modes)
            auto setThroughputStreams = [&]() {
                const std::string key = getDeviceTypeFromName(device) + "_THROUGHPUT_STREAMS";
                if (device_nstreams.count(device)) {
                    // set to user defined value
                    std::vector<std::string> supported_config_keys =
                        core.get_metric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                    if (std::find(supported_config_keys.begin(), supported_config_keys.end(), key) ==
                        supported_config_keys.end()) {
                        throw std::logic_error("Device " + device + " doesn't support config key '" + key + "'! " +
                                               "Please specify -nstreams for correct devices in format  "
                                               "<dev1>:<nstreams1>,<dev2>:<nstreams2>" +
                                               " or via configuration file.");
                    }
                    device_config[key] = device_nstreams.at(device);
                } else if (ov_perf_hint.empty() && !device_config.count(key) && (FLAGS_api == "async")) {
                    slog::warn << "-nstreams default value is determined automatically for " << device
                               << " device. "
                                  "Although the automatic selection usually provides a "
                                  "reasonable performance, "
                                  "but it still may be non-optimal for some cases, for more "
                                  "information look at README."
                               << slog::endl;
                    if (std::string::npos == device.find("MYRIAD"))  // MYRIAD sets the default number of
                                                                     // streams implicitly (without _AUTO)
                        device_config[key] = std::string(getDeviceTypeFromName(device) + "_THROUGHPUT_AUTO");
                }
                if (device_config.count(key))
                    device_nstreams[device] = device_config.at(key);
            };

            if (device.find("CPU") != std::string::npos) {  // CPU supports few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (isFlagSetInCommandLine("nthreads"))
                    device_config[CONFIG_KEY(CPU_THREADS_NUM)] = std::to_string(FLAGS_nthreads);

                if (isFlagSetInCommandLine("enforcebf16"))
                    device_config[CONFIG_KEY(ENFORCE_BF16)] = FLAGS_enforcebf16 ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);

                if (isFlagSetInCommandLine("pin")) {
                    // set to user defined value
                    device_config[CONFIG_KEY(CPU_BIND_THREAD)] = FLAGS_pin;
                } else if (!device_config.count(CONFIG_KEY(CPU_BIND_THREAD))) {
                    if ((device_name.find("MULTI") != std::string::npos) &&
                        (device_name.find("GPU") != std::string::npos)) {
                        slog::warn << "Turn off threads pinning for " << device
                                   << " device since multi-scenario with GPU device is used." << slog::endl;
                        device_config[CONFIG_KEY(CPU_BIND_THREAD)] = CONFIG_VALUE(NO);
                    }
                }

                // for CPU execution, more throughput-oriented execution via streams
                setThroughputStreams();
            } else if (device.find("GPU") != std::string::npos) {
                // for GPU execution, more throughput-oriented execution via streams
                setThroughputStreams();

                if ((device_name.find("MULTI") != std::string::npos) &&
                    (device_name.find("CPU") != std::string::npos)) {
                    slog::warn << "Turn on GPU throttling. Multi-device execution with "
                                  "the CPU + GPU performs best with GPU throttling hint, "
                               << "which releases another CPU thread (that is otherwise "
                                  "used by the GPU driver for active polling)"
                               << slog::endl;
                    device_config[GPU_CONFIG_KEY(PLUGIN_THROTTLE)] = "1";
                }
            } else if (device.find("MYRIAD") != std::string::npos) {
                device_config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
                setThroughputStreams();
            } else if (device.find("GNA") != std::string::npos) {
                if (FLAGS_qb == 8)
                    device_config[GNA_CONFIG_KEY(PRECISION)] = "I8";
                else
                    device_config[GNA_CONFIG_KEY(PRECISION)] = "I16";
            } else {
                std::vector<std::string> supported_config_keys =
                    core.get_metric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                auto supported = [&](const std::string& key) {
                    return std::find(std::begin(supported_config_keys), std::end(supported_config_keys), key) !=
                           std::end(supported_config_keys);
                };
                if (supported(CONFIG_KEY(CPU_THREADS_NUM)) && isFlagSetInCommandLine("nthreads")) {
                    device_config[CONFIG_KEY(CPU_THREADS_NUM)] = std::to_string(FLAGS_nthreads);
                }
                if (supported(CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) && isFlagSetInCommandLine("nstreams")) {
                    device_config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = FLAGS_nstreams;
                }
                if (supported(CONFIG_KEY(CPU_BIND_THREAD)) && isFlagSetInCommandLine("pin")) {
                    device_config[CONFIG_KEY(CPU_BIND_THREAD)] = FLAGS_pin;
                }
            }
        }

        for (auto&& item : config) {
            core.set_config(item.second, item.first);
        }

        size_t batchSize = FLAGS_b;
        ov::element::Type type = ov::element::undefined;
        std::string topology_name = "";
        std::vector<benchmark_app::InputsInfo> app_inputs_info;
        std::string output_name;

        // Takes priority over config from file
        if (!FLAGS_cache_dir.empty()) {
            core.set_config({{CONFIG_KEY(CACHE_DIR), FLAGS_cache_dir}});
        }

        bool isDynamicNetwork = false;

        if (FLAGS_load_from_file && !isNetworkCompiled) {
            next_step();
            slog::info << "Skipping the step for loading network from file" << slog::endl;
            next_step();
            slog::info << "Skipping the step for loading network from file" << slog::endl;
            next_step();
            slog::info << "Skipping the step for loading network from file" << slog::endl;
            auto startTime = Time::now();
            compiledModel = core.compile_model(FLAGS_m, device_name);
            auto duration_ms = double_to_string(get_duration_ms_till_now(startTime));
            slog::info << "Load network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {{"load network time (ms)", duration_ms}});
            app_inputs_info = get_inputs_info(FLAGS_shape,
                                              FLAGS_layout,
                                              batchSize,
                                              FLAGS_data_shape,
                                              inputFiles,
                                              FLAGS_iscale,
                                              FLAGS_imean,
                                              compiledModel.inputs());
            if (batchSize == 0) {
                batchSize = 1;
            }

        } else if (!isNetworkCompiled) {
            // ----------------- 4. Reading the Intermediate Representation network
            // ----------------------------------------
            next_step();

            slog::info << "Loading network files" << slog::endl;

            auto startTime = Time::now();
            auto model = core.read_model(FLAGS_m);
            auto duration_ms = double_to_string(get_duration_ms_till_now(startTime));
            slog::info << "Read network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {{"read network time (ms)", duration_ms}});

            const auto& inputInfo = std::const_pointer_cast<const ov::Model>(model)->inputs();
            if (inputInfo.empty()) {
                throw std::logic_error("no inputs info is provided");
            }

            // ----------------- 5. Resizing network to match image sizes and given
            // batch ----------------------------------
            next_step();
            // Parse input shapes if specified
            bool reshape = false;
            app_inputs_info = get_inputs_info(FLAGS_shape,
                                              FLAGS_layout,
                                              FLAGS_b,
                                              FLAGS_data_shape,
                                              inputFiles,
                                              FLAGS_iscale,
                                              FLAGS_imean,
                                              inputInfo,
                                              reshape);
            if (reshape) {
                benchmark_app::PartialShapes shapes = {};
                for (auto& item : app_inputs_info[0])
                    shapes[item.first] = item.second.partialShape;
                slog::info << "Reshaping network: " << get_shapes_string(shapes) << slog::endl;
                startTime = Time::now();
                model->reshape(shapes);
                duration_ms = double_to_string(get_duration_ms_till_now(startTime));
                slog::info << "Reshape network took " << duration_ms << " ms" << slog::endl;
                if (statistics)
                    statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                               {{"reshape network time (ms)", duration_ms}});
            }

            // ----------------- 6. Configuring inputs and outputs
            // ----------------------------------------------------------------------
            next_step();
            auto preproc = ov::preprocess::PrePostProcessor(model);

            ov::runtime::ConfigMap user_precisions_map;
            if (!FLAGS_iop.empty()) {
                user_precisions_map = parseArgMap(FLAGS_iop);
            }

            const auto input_precision = FLAGS_ip.empty() ? ov::element::undefined : getPrecision2(FLAGS_ip);
            const auto output_precision = FLAGS_op.empty() ? ov::element::undefined : getPrecision2(FLAGS_op);

            const auto& inputs = model->inputs();
            for (int i = 0; i < inputs.size(); i++) {
                const auto& item = inputs[i];
                auto iop_precision = ov::element::undefined;
                auto type_to_set = ov::element::undefined;
                std::string name;
                try {
                    // Some tensors might have no names, get_any_name will throw exception in that case.
                    // -iop option will not work for those tensors.
                    name = item.get_any_name();
                    iop_precision = getPrecision2(user_precisions_map.at(item.get_any_name()));
                } catch (...) {
                }

                if (iop_precision != ov::element::undefined) {
                    type_to_set = iop_precision;
                } else if (input_precision != ov::element::undefined) {
                    type_to_set = input_precision;
                } else if (!name.empty() && app_inputs_info[0].at(name).is_image()) {
                    // image input, set U8
                    type_to_set = ov::element::u8;
                }

                auto& in = preproc.input(item.get_index());
                if (type_to_set != ov::element::undefined) {
                    in.tensor().set_element_type(type_to_set);

                    if (!name.empty()) {
                        for (auto& info : app_inputs_info) {
                            info.at(name).type = type_to_set;
                        }
                    }
                    // Explicitly set inputs layout.
                    in.model().set_layout(app_inputs_info[0].at(name).layout);
                }
            }

            const auto& outs = model->outputs();
            for (int i = 0; i < outs.size(); i++) {
                const auto& item = outs[i];
                auto iop_precision = ov::element::undefined;
                try {
                    // Some tensors might have no names, get_any_name will throw exception in that case.
                    // -iop option will not work for those tensors.
                    iop_precision = getPrecision2(user_precisions_map.at(item.get_any_name()));
                } catch (...) {
                }

                if (iop_precision != ov::element::undefined) {
                    preproc.output(i).tensor().set_element_type(iop_precision);
                } else if (output_precision != ov::element::undefined) {
                    preproc.output(i).tensor().set_element_type(output_precision);
                }
            }

            model = preproc.build();

            // Check if network has dynamic shapes
            auto input_info = app_inputs_info[0];
            isDynamicNetwork = std::any_of(input_info.begin(),
                                           input_info.end(),
                                           [](const std::pair<std::string, benchmark_app::InputInfo>& i) {
                                               return i.second.partialShape.is_dynamic();
                                           });

            topology_name = model->get_friendly_name();

            // Calculate batch size according to provided layout and shapes (static case)
            if (!isDynamicNetwork && app_inputs_info.size()) {
                batchSize = get_batch_size(app_inputs_info.front());

                slog::info << "Network batch size: " << batchSize << slog::endl;
            } else if (batchSize == 0) {
                batchSize = 1;
            }

            printInputAndOutputsInfoShort(*model);
            // ----------------- 7. Loading the model to the device
            // --------------------------------------------------------
            next_step();
            startTime = Time::now();
            compiledModel = core.compile_model(model, device_name);
            duration_ms = double_to_string(get_duration_ms_till_now(startTime));
            slog::info << "Load network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {{"load network time (ms)", duration_ms}});
        } else {
            next_step();
            slog::info << "Skipping the step for compiled network" << slog::endl;
            next_step();
            slog::info << "Skipping the step for compiled network" << slog::endl;
            next_step();
            slog::info << "Skipping the step for compiled network" << slog::endl;
            // ----------------- 7. Loading the model to the device
            // --------------------------------------------------------
            next_step();
            auto startTime = Time::now();

            std::ifstream modelStream(FLAGS_m, std::ios_base::binary | std::ios_base::in);
            if (!modelStream.is_open()) {
                throw std::runtime_error("Cannot open model file " + FLAGS_m);
            }
            compiledModel = core.import_model(modelStream, device_name, {});
            modelStream.close();

            auto duration_ms = double_to_string(get_duration_ms_till_now(startTime));
            slog::info << "Import network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {{"import network time (ms)", duration_ms}});

            app_inputs_info = get_inputs_info(FLAGS_shape,
                                              FLAGS_layout,
                                              FLAGS_b,
                                              FLAGS_data_shape,
                                              inputFiles,
                                              FLAGS_iscale,
                                              FLAGS_imean,
                                              compiledModel.inputs());
            if (batchSize == 0) {
                batchSize = 1;
            }
        }

        if (isDynamicNetwork && FLAGS_api == "sync") {
            throw std::logic_error("Benchmarking of the model with dynamic shapes is available for async API only."
                                   "Please use -api async -nstreams 1 -nireq 1 to emulate sync behavior");
        }

        // Defining of benchmark mode
        // for static models inference only mode is used as default one
        bool inferenceOnly = FLAGS_inference_only;
        if (isDynamicNetwork) {
            if (isFlagSetInCommandLine("inference_only") && inferenceOnly && app_inputs_info.size() != 1) {
                throw std::logic_error(
                    "Dynamic models with different input data shapes must be benchmarked only in full mode.");
            }
            inferenceOnly = isFlagSetInCommandLine("inference_only") && inferenceOnly && app_inputs_info.size() == 1;
        }

        // ----------------- 8. Querying optimal runtime parameters
        // -----------------------------------------------------
        next_step();
        // output of the actual settings that the device selected
        for (const auto& device : devices) {
            std::vector<std::string> supported_config_keys = core.get_metric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            slog::info << "Device: " << device << slog::endl;
            for (const auto& cfg : supported_config_keys) {
                try {
                    slog::info << "  {" << cfg << " , " << compiledModel.get_config(cfg).as<std::string>();
                    slog::info << " }" << slog::endl;
                } catch (...) {
                };
            }
        }

        // Update number of streams
        for (auto&& ds : device_nstreams) {
            const std::string key = getDeviceTypeFromName(ds.first) + "_THROUGHPUT_STREAMS";
            device_nstreams[ds.first] = core.get_config(ds.first, key).as<std::string>();
        }

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if (nireq == 0) {
            if (FLAGS_api == "sync") {
                nireq = 1;
            } else {
                std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
                try {
                    nireq = compiledModel.get_metric(key).as<unsigned int>();
                } catch (const std::exception& ex) {
                    IE_THROW() << "Every device used with the benchmark_app should "
                               << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS metric. "
                               << "Failed to query the metric for the " << device_name << " with error:" << ex.what();
                }
            }
        }

        // Iteration limit
        uint32_t niter = FLAGS_niter;
        size_t shape_groups_num = app_inputs_info.size();
        if ((niter > 0) && (FLAGS_api == "async")) {
            if (shape_groups_num > nireq) {
                niter = ((niter + shape_groups_num - 1) / shape_groups_num) * shape_groups_num;
                if (FLAGS_niter != niter) {
                    slog::warn << "Number of iterations was aligned by data shape groups number from " << FLAGS_niter
                               << " to " << niter << " using number of possible input shapes " << shape_groups_num
                               << slog::endl;
                }
            } else {
                niter = ((niter + nireq - 1) / nireq) * nireq;
                if (FLAGS_niter != niter) {
                    slog::warn << "Number of iterations was aligned by request number from " << FLAGS_niter << " to "
                               << niter << " using number of requests " << nireq << slog::endl;
                }
            }
        }

        // Time limit
        uint32_t duration_seconds = 0;
        if (FLAGS_t != 0) {
            // time limit
            duration_seconds = FLAGS_t;
        } else if (FLAGS_niter == 0) {
            // default time limit
            duration_seconds = device_default_device_duration_in_seconds(device_name);
        }
        uint64_t duration_nanoseconds = get_duration_in_nanoseconds(duration_seconds);

        if (statistics) {
            statistics->add_parameters(
                StatisticsReport::Category::RUNTIME_CONFIG,
                {
                    {"benchmark mode", inferenceOnly ? "inference only" : "full"},
                    {"topology", topology_name},
                    {"target device", device_name},
                    {"API", FLAGS_api},
                    {"precision", std::string(type.get_type_name())},
                    {"batch size", std::to_string(batchSize)},
                    {"number of iterations", std::to_string(niter)},
                    {"number of parallel infer requests", std::to_string(nireq)},
                    {"duration (ms)", std::to_string(get_duration_in_milliseconds(duration_seconds))},
                });
            for (auto& nstreams : device_nstreams) {
                std::stringstream ss;
                ss << "number of " << nstreams.first << " streams";
                statistics->add_parameters(StatisticsReport::Category::RUNTIME_CONFIG,
                                           {
                                               {ss.str(), nstreams.second},
                                           });
            }
        }

        // ----------------- 9. Creating infer requests and filling input blobs
        // ----------------------------------------
        next_step();

        InferRequestsQueue inferRequestsQueue(compiledModel, nireq, app_inputs_info.size(), FLAGS_pcseq);

        bool inputHasName = false;
        if (inputFiles.size() > 0) {
            inputHasName = inputFiles.begin()->first != "";
        }
        bool newInputType = isDynamicNetwork || inputHasName;
        // create vector to store remote input blobs buffer
        std::vector<::gpu::BufferType> clInputsBuffer;
        bool useGpuMem = false;

        std::map<std::string, ov::runtime::TensorVector> inputsData;
        if (isFlagSetInCommandLine("use_device_mem")) {
            if (device_name.find("GPU") == 0) {
                inputsData =
                    ::gpu::get_remote_input_tensors(inputFiles, app_inputs_info, compiledModel, clInputsBuffer);
                useGpuMem = true;
            } else if (device_name.find("CPU") == 0) {
                if (newInputType) {
                    inputsData = get_tensors(inputFiles, app_inputs_info);
                } else {
                    inputsData = get_tensors_static_case(
                        inputFiles.empty() ? std::vector<std::string>{} : inputFiles.begin()->second,
                        batchSize,
                        app_inputs_info[0],
                        nireq);
                }
            } else {
                IE_THROW() << "Requested device doesn't support `use_device_mem` option.";
            }
        } else {
            if (newInputType) {
                inputsData = get_tensors(inputFiles, app_inputs_info);
            } else {
                inputsData = get_tensors_static_case(
                    inputFiles.empty() ? std::vector<std::string>{} : inputFiles.begin()->second,
                    batchSize,
                    app_inputs_info[0],
                    nireq);
            }
        }
        // ----------------- 10. Measuring performance
        // ------------------------------------------------------------------
        size_t progressCnt = 0;
        size_t progressBarTotalCount = progressBarDefaultTotalCount;
        size_t iteration = 0;

        std::stringstream ss;
        ss << "Start inference " << FLAGS_api << "hronously";
        if (FLAGS_api == "async") {
            if (!ss.str().empty()) {
                ss << ", ";
            }
            ss << nireq << " inference requests";
            std::stringstream device_ss;
            for (auto& nstreams : device_nstreams) {
                if (!device_ss.str().empty()) {
                    device_ss << ", ";
                }
                device_ss << nstreams.second << " streams for " << nstreams.first;
            }
            if (!device_ss.str().empty()) {
                ss << " using " << device_ss.str();
            }
        }
        ss << ", limits: ";
        if (duration_seconds > 0) {
            ss << get_duration_in_milliseconds(duration_seconds) << " ms duration";
        }
        if (niter != 0) {
            if (duration_seconds == 0) {
                progressBarTotalCount = niter;
            }
            if (duration_seconds > 0) {
                ss << ", ";
            }
            ss << niter << " iterations";
        }

        next_step(ss.str());

        if (inferenceOnly) {
            slog::info << "BENCHMARK IS IN INFERENCE ONLY MODE." << slog::endl;
            slog::info << "Input blobs will be filled once before performance measurements." << slog::endl;
        } else {
            slog::info << "BENCHMARK IS IN FULL MODE." << slog::endl;
            slog::info << "Inputs setup stage will be included in performance measurements." << slog::endl;
        }

        // copy prepared data straight into inferRequest->getTensor()
        // for inference only mode
        if (inferenceOnly) {
            if (nireq < inputsData.begin()->second.size())
                slog::warn << "Only " << nireq << " test configs will be used." << slog::endl;
            size_t i = 0;
            for (auto& inferRequest : inferRequestsQueue.requests) {
                auto inputs = app_inputs_info[i % app_inputs_info.size()];
                for (auto& item : inputs) {
                    auto inputName = item.first;
                    const auto& inputTensor = inputsData.at(inputName)[i % inputsData.at(inputName).size()];
                    // for remote blobs setTensor is used, they are already allocated on the device
                    if (useGpuMem) {
                        inferRequest->set_tensor(inputName, inputTensor);
                    } else {
                        auto requestTensor = inferRequest->get_tensor(inputName);
                        if (isDynamicNetwork) {
                            requestTensor.set_shape(inputTensor.get_shape());
                        }
                        copy_tensor_data(requestTensor, inputTensor);
                    }
                }

                if (useGpuMem) {
                    auto outputTensors =
                        ::gpu::get_remote_output_tensors(compiledModel, inferRequest->get_output_cl_buffer());
                    for (auto& output : compiledModel.outputs()) {
                        inferRequest->set_tensor(output.get_any_name(), outputTensors[output.get_any_name()]);
                    }
                }
                ++i;
            }
        }

        // warming up - out of scope
        auto inferRequest = inferRequestsQueue.get_idle_request();
        if (!inferRequest) {
            IE_THROW() << "No idle Infer Requests!";
        }

        if (!inferenceOnly) {
            auto inputs = app_inputs_info[0];

            for (auto& item : inputs) {
                auto inputName = item.first;
                const auto& data = inputsData.at(inputName)[0];
                inferRequest->set_tensor(inputName, data);
            }

            if (useGpuMem) {
                auto outputTensors =
                    ::gpu::get_remote_output_tensors(compiledModel, inferRequest->get_output_cl_buffer());
                for (auto& output : compiledModel.outputs()) {
                    inferRequest->set_tensor(output.get_any_name(), outputTensors[output.get_any_name()]);
                }
            }
        }

        if (FLAGS_api == "sync") {
            inferRequest->infer();
        } else {
            inferRequest->start_async();
        }

        inferRequestsQueue.wait_all();

        auto duration_ms = double_to_string(inferRequestsQueue.get_latencies()[0]);
        slog::info << "First inference took " << duration_ms << " ms" << slog::endl;

        if (statistics) {
            statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                       {{"first inference time (ms)", duration_ms}});
        }
        inferRequestsQueue.reset_times();

        size_t processedFramesN = 0;
        auto startTime = Time::now();
        auto execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

        /** Start inference & calculate performance **/
        /** to align number if iterations to guarantee that last infer requests are
         * executed in the same conditions **/
        ProgressBar progressBar(progressBarTotalCount, FLAGS_stream_output, FLAGS_progress);
        while ((niter != 0LL && iteration < niter) ||
               (duration_nanoseconds != 0LL && (uint64_t)execTime < duration_nanoseconds) ||
               (FLAGS_api == "async" && iteration % nireq != 0)) {
            inferRequest = inferRequestsQueue.get_idle_request();
            if (!inferRequest) {
                IE_THROW() << "No idle Infer Requests!";
            }

            if (!inferenceOnly) {
                auto inputs = app_inputs_info[iteration % app_inputs_info.size()];

                if (FLAGS_pcseq) {
                    inferRequest->set_latency_group_id(iteration % app_inputs_info.size());
                }

                if (isDynamicNetwork) {
                    batchSize = get_batch_size(inputs);
                    if (!std::any_of(inputs.begin(),
                                     inputs.end(),
                                     [](const std::pair<const std::string, benchmark_app::InputInfo>& info) {
                                         return ov::layout::has_batch(info.second.layout);
                                     })) {
                        slog::warn
                            << "No batch dimension was found, asssuming batch to be 1. Beware: this might affect "
                               "FPS calculation."
                            << slog::endl;
                    }
                }

                for (auto& item : inputs) {
                    auto inputName = item.first;
                    const auto& data = inputsData.at(inputName)[iteration % inputsData.at(inputName).size()];
                    inferRequest->set_tensor(inputName, data);
                }

                if (useGpuMem) {
                    auto outputTensors =
                        ::gpu::get_remote_output_tensors(compiledModel, inferRequest->get_output_cl_buffer());
                    for (auto& output : compiledModel.outputs()) {
                        inferRequest->set_tensor(output.get_any_name(), outputTensors[output.get_any_name()]);
                    }
                }
            }

            if (FLAGS_api == "sync") {
                inferRequest->infer();
            } else {
                // As the inference request is currently idle, the wait() adds no
                // additional overhead (and should return immediately). The primary
                // reason for calling the method is exception checking/re-throwing.
                // Callback, that governs the actual execution can handle errors as
                // well, but as it uses just error codes it has no details like ‘what()’
                // method of `std::exception` So, rechecking for any exceptions here.
                inferRequest->wait();
                inferRequest->start_async();
            }
            ++iteration;

            execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();
            processedFramesN += batchSize;

            if (niter > 0) {
                progressBar.add_progress(1);
            } else {
                // calculate how many progress intervals are covered by current
                // iteration. depends on the current iteration time and time of each
                // progress interval. Previously covered progress intervals must be
                // skipped.
                auto progressIntervalTime = duration_nanoseconds / progressBarTotalCount;
                size_t newProgress = execTime / progressIntervalTime - progressCnt;
                progressBar.add_progress(newProgress);
                progressCnt += newProgress;
            }
        }

        // wait the latest inference executions
        inferRequestsQueue.wait_all();

        LatencyMetrics generalLatency(inferRequestsQueue.get_latencies());
        std::vector<LatencyMetrics> groupLatencies = {};
        if (FLAGS_pcseq && app_inputs_info.size() > 1) {
            for (auto lats : inferRequestsQueue.get_latency_groups()) {
                groupLatencies.push_back(LatencyMetrics(lats));
            }
        }

        double totalDuration = inferRequestsQueue.get_duration_in_milliseconds();
        double fps = (FLAGS_api == "sync") ? batchSize * 1000.0 / generalLatency.percentile(FLAGS_latency_percentile)
                                           : 1000.0 * processedFramesN / totalDuration;

        if (statistics) {
            statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                       {
                                           {"total execution time (ms)", double_to_string(totalDuration)},
                                           {"total number of iterations", std::to_string(iteration)},
                                       });
            if (device_name.find("MULTI") == std::string::npos) {
                std::string latency_label;
                if (FLAGS_latency_percentile == 50) {
                    latency_label = "Median latency (ms)";
                } else {
                    latency_label = "latency (" + std::to_string(FLAGS_latency_percentile) + " percentile) (ms)";
                }
                statistics->add_parameters(
                    StatisticsReport::Category::EXECUTION_RESULTS,
                    {
                        {latency_label, double_to_string(generalLatency.percentile(FLAGS_latency_percentile))},
                    });
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {
                                               {"Average latency (ms)", double_to_string(generalLatency.average())},
                                           });
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {
                                               {"Min latency (ms)", double_to_string(generalLatency.min())},
                                           });
                statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                           {
                                               {"Max latency (ms)", double_to_string(generalLatency.max())},
                                           });

                if (FLAGS_pcseq && app_inputs_info.size() > 1) {
                    statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                               {
                                                   {"Latency for each data shape group:", ""},
                                               });
                    for (size_t i = 0; i < app_inputs_info.size(); ++i) {
                        std::string data_shapes_string = "";
                        data_shapes_string += std::to_string(i + 1) + ". ";
                        for (auto& item : app_inputs_info[i]) {
                            data_shapes_string += item.first + " : " + get_shape_string(item.second.dataShape) + " ";
                        }
                        statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                                   {
                                                       {data_shapes_string, ""},
                                                   });
                        statistics->add_parameters(
                            StatisticsReport::Category::EXECUTION_RESULTS,
                            {
                                {latency_label,
                                 double_to_string(groupLatencies[i].percentile(FLAGS_latency_percentile))},
                            });
                        statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                                   {
                                                       {"Average (ms)", double_to_string(groupLatencies[i].average())},
                                                   });
                        statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                                   {
                                                       {"Min (ms)", double_to_string(groupLatencies[i].min())},
                                                   });
                        statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                                   {
                                                       {"Max (ms)", double_to_string(groupLatencies[i].max())},
                                                   });
                    }
                }
            }
            statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                       {{"throughput", double_to_string(fps)}});
        }
        progressBar.finish();

        // ----------------- 11. Dumping statistics report
        // -------------------------------------------------------------
        next_step();

        if (!FLAGS_dump_config.empty()) {
            dump_config(FLAGS_dump_config, config);
            slog::info << "Inference Engine configuration settings were dumped to " << FLAGS_dump_config << slog::endl;
        }

        if (!FLAGS_exec_graph_path.empty()) {
            try {
                std::string fileName = fileNameNoExt(FLAGS_exec_graph_path);
                ov::pass::Serialize serializer(fileName + ".xml", fileName + ".bin");
                serializer.run_on_model(std::const_pointer_cast<ov::Model>(compiledModel.get_runtime_model()));
                slog::info << "executable graph is stored to " << FLAGS_exec_graph_path << slog::endl;
            } catch (const std::exception& ex) {
                slog::err << "Can't get executable graph: " << ex.what() << slog::endl;
            }
        }

        if (perf_counts) {
            std::vector<std::vector<ov::runtime::ProfilingInfo>> perfCounts;
            for (size_t ireq = 0; ireq < nireq; ireq++) {
                auto reqPerfCounts = inferRequestsQueue.requests[ireq]->get_performance_counts();
                if (FLAGS_pc) {
                    slog::info << "Performance counts for " << ireq << "-th infer request:" << slog::endl;
                    printPerformanceCounts(reqPerfCounts, std::cout, getFullDeviceName(core, FLAGS_d), false);
                }
                perfCounts.push_back(reqPerfCounts);
            }
            if (statistics) {
                statistics->dump_performance_counters(perfCounts);
            }
        }

        if (statistics)
            statistics->dump();

        // Performance metrics report
        slog::info << "Count:      " << iteration << " iterations" << slog::endl;
        slog::info << "Duration:   " << double_to_string(totalDuration) << " ms" << slog::endl;
        if (device_name.find("MULTI") == std::string::npos) {
            slog::info << "Latency: " << slog::endl;
            generalLatency.log_total(FLAGS_latency_percentile);

            if (FLAGS_pcseq && app_inputs_info.size() > 1) {
                slog::info << "Latency for each data shape group:" << slog::endl;
                for (size_t i = 0; i < app_inputs_info.size(); ++i) {
                    slog::info << (i + 1) << ".";
                    for (auto& item : app_inputs_info[i]) {
                        std::stringstream input_shape;
                        auto shape = item.second.dataShape;
                        std::copy(shape.begin(), shape.end() - 1, std::ostream_iterator<size_t>(input_shape, ","));
                        input_shape << shape.back();
                        slog::info << " " << item.first << " : " << get_shape_string(item.second.dataShape);
                    }
                    slog::info << slog::endl;

                    groupLatencies[i].log_total(FLAGS_latency_percentile);
                }
            }
        }
        slog::info << "Throughput: " << double_to_string(fps) << " FPS" << slog::endl;

    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;

        if (statistics) {
            statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                       {
                                           {"error", ex.what()},
                                       });
            statistics->dump();
        }

        return 3;
    }

    return 0;
}
