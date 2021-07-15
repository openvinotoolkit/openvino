// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <gna/gna_config.hpp>
#include <gpu/gpu_config.hpp>
#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <utility>
#include <vector>
#include <vpu/vpu_plugin_config.hpp>

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "inputs_filling.hpp"
#include "progress_bar.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"

using namespace InferenceEngine;

static const size_t progressBarDefaultTotalCount = 1000;

uint64_t getDurationInMilliseconds(uint32_t duration) {
    return duration * 1000LL;
}

uint64_t getDurationInNanoseconds(uint32_t duration) {
    return duration * 1000000000LL;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validating input
    // arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_api != "async" && FLAGS_api != "sync") {
        throw std::logic_error("Incorrect API. Please set -api option to `sync` or `async` value.");
    }

    if (!FLAGS_report_type.empty() && FLAGS_report_type != noCntReport && FLAGS_report_type != averageCntReport && FLAGS_report_type != detailedCntReport) {
        std::string err = "only " + std::string(noCntReport) + "/" + std::string(averageCntReport) + "/" + std::string(detailedCntReport) +
                          " report types are supported (invalid -report_type option value)";
        throw std::logic_error(err);
    }

    if ((FLAGS_report_type == averageCntReport) && ((FLAGS_d.find("MULTI") != std::string::npos))) {
        throw std::logic_error("only " + std::string(detailedCntReport) + " report type is supported for MULTI device");
    }

    bool isNetworkCompiled = fileExt(FLAGS_m) == "blob";
    bool isPrecisionSet = !(FLAGS_ip.empty() && FLAGS_op.empty() && FLAGS_iop.empty());
    if (isNetworkCompiled && isPrecisionSet) {
        std::string err = std::string("Cannot set precision for a compiled network. ") + std::string("Please re-compile your network with required precision "
                                                                                                     "using compile_tool");

        throw std::logic_error(err);
    }
    return true;
}

static void next_step(const std::string additional_info = "") {
    static size_t step_id = 0;
    static const std::map<size_t, std::string> step_names = {{1, "Parsing and validating input arguments"},
                                                             {2, "Loading Inference Engine"},
                                                             {3, "Setting device configuration"},
                                                             {4, "Reading network files"},
                                                             {5, "Resizing network to match image sizes and given batch"},
                                                             {6, "Configuring input of the model"},
                                                             {7, "Loading the model to the device"},
                                                             {8, "Setting optimal runtime parameters"},
                                                             {9, "Creating infer requests and filling input blobs with images"},
                                                             {10, "Measuring performance"},
                                                             {11, "Dumping statistics report"}};

    step_id++;
    if (step_names.count(step_id) == 0)
        IE_THROW() << "Step ID " << step_id << " is out of total steps number " << step_names.size();

    std::cout << "[Step " << step_id << "/" << step_names.size() << "] " << step_names.at(step_id)
              << (additional_info.empty() ? "" : " (" + additional_info + ")") << std::endl;
}

template <typename T>
T getMedianValue(const std::vector<T>& vec) {
    std::vector<T> sortedVec(vec);
    std::sort(sortedVec.begin(), sortedVec.end());
    return (sortedVec.size() % 2 != 0) ? sortedVec[sortedVec.size() / 2ULL]
                                       : (sortedVec[sortedVec.size() / 2ULL] + sortedVec[sortedVec.size() / 2ULL - 1ULL]) / static_cast<T>(2.0);
}

/**
 * @brief The entry point of the benchmark application
 */
int main(int argc, char* argv[]) {
    std::shared_ptr<StatisticsReport> statistics;
    try {
        ExecutableNetwork exeNetwork;

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
            statistics = std::make_shared<StatisticsReport>(StatisticsReport::Config {FLAGS_report_type, FLAGS_report_folder});
            statistics->addParameters(StatisticsReport::Category::COMMAND_LINE_PARAMETERS, command_line_arguments);
        }
        auto isFlagSetInCommandLine = [&command_line_arguments](const std::string& name) {
            return (std::find_if(command_line_arguments.begin(), command_line_arguments.end(), [name](const std::pair<std::string, std::string>& p) {
                        return p.first == name;
                    }) != command_line_arguments.end());
        };

        std::string device_name = FLAGS_d;

        // Parse devices
        auto devices = parseDevices(device_name);

        // Parse nstreams per device
        std::map<std::string, std::string> device_nstreams = parseNStreamsValuePerDevice(devices, FLAGS_nstreams);

        // Load device config file if specified
        std::map<std::string, std::map<std::string, std::string>> config;
#ifdef USE_OPENCV
        if (!FLAGS_load_config.empty()) {
            load_config(FLAGS_load_config, config);
        }
#endif
        /** This vector stores paths to the processed images **/
        std::vector<std::string> inputFiles;
        parseInputFilesArguments(inputFiles);

        // ----------------- 2. Loading the Inference Engine
        // -----------------------------------------------------------
        next_step();

        Core ie;
        if (FLAGS_d.find("CPU") != std::string::npos && !FLAGS_l.empty()) {
            // CPU (MKLDNN) extensions is loaded as a shared library and passed as a
            // pointer to base extension
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>(FLAGS_l);
            ie.AddExtension(extension_ptr);
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
            ie.SetConfig({{CONFIG_KEY(CONFIG_FILE), ext}}, "GPU");
            slog::info << "GPU extensions is loaded " << ext << slog::endl;
        }

        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(device_name) << std::endl;

        // ----------------- 3. Setting device configuration
        // -----------------------------------------------------------
        next_step();
        std::string ov_perf_hint;
        if (FLAGS_hint == "throughput" || FLAGS_hint == "THROUGHPUT" || FLAGS_hint == "tput")
            ov_perf_hint = CONFIG_VALUE(THROUGHPUT);
        else if (FLAGS_hint == "latency" || FLAGS_hint == "LATENCY")
            ov_perf_hint = CONFIG_VALUE(LATENCY);
        else if (!FLAGS_hint.empty())
            throw std::logic_error("Performance hint " + ov_perf_hint + " is not recognized!");

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
            } else if (device_config.count(CONFIG_KEY(PERF_COUNT)) && (device_config.at(CONFIG_KEY(PERF_COUNT)) == "YES")) {
                slog::warn << "Performance counters for " << device << " device is turned on. To print results use -pc option." << slog::endl;
            } else if (FLAGS_report_type == detailedCntReport || FLAGS_report_type == averageCntReport) {
                slog::warn << "Turn on performance counters for " << device << " device since report type is " << FLAGS_report_type << "." << slog::endl;
                device_config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
            } else if (!FLAGS_exec_graph_path.empty()) {
                slog::warn << "Turn on performance counters for " << device << " device due to execution graph dumping." << slog::endl;
                device_config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
            } else {
                // set to default value
                device_config[CONFIG_KEY(PERF_COUNT)] = FLAGS_pc ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
            }
            perf_counts = (device_config.at(CONFIG_KEY(PERF_COUNT)) == CONFIG_VALUE(YES)) ? true : perf_counts;

            // the rest are individual per-device settings (overriding the values set with perf modes)
            auto setThroughputStreams = [&]() {
                const std::string key = device + "_THROUGHPUT_STREAMS";
                if (device_nstreams.count(device)) {
                    // set to user defined value
                    std::vector<std::string> supported_config_keys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                    if (std::find(supported_config_keys.begin(), supported_config_keys.end(), key) == supported_config_keys.end()) {
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
                        device_config[key] = std::string(device + "_THROUGHPUT_AUTO");
                }
                if (device_config.count(key))
                    device_nstreams[device] = device_config.at(key);
            };

            if (device == "CPU") {  // CPU supports few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (isFlagSetInCommandLine("nthreads"))
                    device_config[CONFIG_KEY(CPU_THREADS_NUM)] = std::to_string(FLAGS_nthreads);

                if (isFlagSetInCommandLine("enforcebf16"))
                    device_config[CONFIG_KEY(ENFORCE_BF16)] = FLAGS_enforcebf16 ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);

                if (isFlagSetInCommandLine("pin")) {
                    // set to user defined value
                    device_config[CONFIG_KEY(CPU_BIND_THREAD)] = FLAGS_pin;
                } else if (!device_config.count(CONFIG_KEY(CPU_BIND_THREAD))) {
                    if ((device_name.find("MULTI") != std::string::npos) && (device_name.find("GPU") != std::string::npos)) {
                        slog::warn << "Turn off threads pinning for " << device << " device since multi-scenario with GPU device is used." << slog::endl;
                        device_config[CONFIG_KEY(CPU_BIND_THREAD)] = CONFIG_VALUE(NO);
                    }
                }

                // for CPU execution, more throughput-oriented execution via streams
                setThroughputStreams();
            } else if (device == ("GPU")) {
                // for GPU execution, more throughput-oriented execution via streams
                setThroughputStreams();

                if ((device_name.find("MULTI") != std::string::npos) && (device_name.find("CPU") != std::string::npos)) {
                    slog::warn << "Turn on GPU trottling. Multi-device execution with "
                                  "the CPU + GPU performs best with GPU trottling hint,"
                               << "which releases another CPU thread (that is otherwise "
                                  "used by the GPU driver for active polling)"
                               << slog::endl;
                    device_config[GPU_CONFIG_KEY(PLUGIN_THROTTLE)] = "1";
                }
            } else if (device == "MYRIAD") {
                device_config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
                setThroughputStreams();
            } else if (device == "GNA") {
                if (FLAGS_qb == 8)
                    device_config[GNA_CONFIG_KEY(PRECISION)] = "I8";
                else
                    device_config[GNA_CONFIG_KEY(PRECISION)] = "I16";

                if (isFlagSetInCommandLine("nthreads"))
                    device_config[GNA_CONFIG_KEY(LIB_N_THREADS)] = std::to_string(FLAGS_nthreads);
            } else {
                std::vector<std::string> supported_config_keys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                auto supported = [&](const std::string& key) {
                    return std::find(std::begin(supported_config_keys), std::end(supported_config_keys), key) != std::end(supported_config_keys);
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
            ie.SetConfig(item.second, item.first);
        }

        auto double_to_string = [](const double number) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << number;
            return ss.str();
        };
        auto get_total_ms_time = [](Time::time_point& startTime) {
            return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
        };

        size_t batchSize = FLAGS_b;
        Precision precision = Precision::UNSPECIFIED;
        std::string topology_name = "";
        benchmark_app::InputsInfo app_inputs_info;
        std::string output_name;

        // Takes priority over config from file
        if (!FLAGS_cache_dir.empty()) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), FLAGS_cache_dir}});
        }

        if (FLAGS_load_from_file && !isNetworkCompiled) {
            next_step();
            slog::info << "Skipping the step for loading network from file" << slog::endl;
            next_step();
            slog::info << "Skipping the step for loading network from file" << slog::endl;
            next_step();
            slog::info << "Skipping the step for loading network from file" << slog::endl;
            auto startTime = Time::now();
            exeNetwork = ie.LoadNetwork(FLAGS_m, device_name);
            auto duration_ms = double_to_string(get_total_ms_time(startTime));
            slog::info << "Load network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"load network time (ms)", duration_ms}});
            if (batchSize == 0) {
                batchSize = 1;
            }
        } else if (!isNetworkCompiled) {
            // ----------------- 4. Reading the Intermediate Representation network
            // ----------------------------------------
            next_step();

            slog::info << "Loading network files" << slog::endl;

            auto startTime = Time::now();
            CNNNetwork cnnNetwork = ie.ReadNetwork(FLAGS_m);
            auto duration_ms = double_to_string(get_total_ms_time(startTime));
            slog::info << "Read network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"read network time (ms)", duration_ms}});

            const InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
            if (inputInfo.empty()) {
                throw std::logic_error("no inputs info is provided");
            }

            // ----------------- 5. Resizing network to match image sizes and given
            // batch ----------------------------------
            next_step();
            batchSize = cnnNetwork.getBatchSize();
            // Parse input shapes if specified
            bool reshape = false;
            app_inputs_info = getInputsInfo<InputInfo::Ptr>(FLAGS_shape, FLAGS_layout, FLAGS_b, inputInfo, reshape);
            if (reshape) {
                InferenceEngine::ICNNNetwork::InputShapes shapes = {};
                for (auto& item : app_inputs_info)
                    shapes[item.first] = item.second.shape;
                slog::info << "Reshaping network: " << getShapesString(shapes) << slog::endl;
                startTime = Time::now();
                cnnNetwork.reshape(shapes);
                duration_ms = double_to_string(get_total_ms_time(startTime));
                slog::info << "Reshape network took " << duration_ms << " ms" << slog::endl;
                if (statistics)
                    statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"reshape network time (ms)", duration_ms}});
            }
            // use batch size according to provided layout and shapes
            batchSize = (!FLAGS_layout.empty()) ? getBatchSize(app_inputs_info) : cnnNetwork.getBatchSize();

            topology_name = cnnNetwork.getName();
            slog::info << (FLAGS_b != 0 ? "Network batch size was changed to: " : "Network batch size: ") << batchSize << slog::endl;

            // ----------------- 6. Configuring inputs and outputs
            // ----------------------------------------------------------------------
            next_step();

            processPrecision(cnnNetwork, FLAGS_ip, FLAGS_op, FLAGS_iop);
            for (auto& item : cnnNetwork.getInputsInfo()) {
                // if precision for input set by user, then set it to app_inputs
                // if it an image, set U8
                if (!FLAGS_ip.empty() || FLAGS_iop.find(item.first) != std::string::npos) {
                    app_inputs_info.at(item.first).precision = item.second->getPrecision();
                } else if (app_inputs_info.at(item.first).isImage()) {
                    app_inputs_info.at(item.first).precision = Precision::U8;
                    item.second->setPrecision(app_inputs_info.at(item.first).precision);
                }
            }

            printInputAndOutputsInfo(cnnNetwork);
            // ----------------- 7. Loading the model to the device
            // --------------------------------------------------------
            next_step();
            startTime = Time::now();
            exeNetwork = ie.LoadNetwork(cnnNetwork, device_name);
            duration_ms = double_to_string(get_total_ms_time(startTime));
            slog::info << "Load network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"load network time (ms)", duration_ms}});

            if (!ov_perf_hint.empty()) {
                std::cout << "PERFORMANCE_HINT: " << ov_perf_hint << std::endl;
                // output of the actual settings that the mode produces (debugging)
                for (const auto& device : devices) {
                    std::vector<std::string> supported_config_keys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                    std::cout << "Device: " << device << std::endl;
                    for (const auto& cfg : supported_config_keys) {
                        std::cout << "  {" << cfg << " , " << exeNetwork.GetConfig(cfg).as<std::string>() << " }" << std::endl;
                    }
                }
            }
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
            exeNetwork = ie.ImportNetwork(FLAGS_m, device_name, {});
            auto duration_ms = double_to_string(get_total_ms_time(startTime));
            slog::info << "Import network took " << duration_ms << " ms" << slog::endl;
            if (statistics)
                statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"import network time (ms)", duration_ms}});
            app_inputs_info = getInputsInfo<InputInfo::CPtr>(FLAGS_shape, FLAGS_layout, FLAGS_b, exeNetwork.GetInputsInfo());
            if (batchSize == 0) {
                batchSize = 1;
            }
        }
        // ----------------- 8. Setting optimal runtime parameters
        // -----------------------------------------------------
        next_step();

        // Update number of streams
        for (auto&& ds : device_nstreams) {
            const std::string key = ds.first + "_THROUGHPUT_STREAMS";
            device_nstreams[ds.first] = ie.GetConfig(ds.first, key).as<std::string>();
        }

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if (nireq == 0) {
            if (FLAGS_api == "sync") {
                nireq = 1;
            } else {
                std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
                try {
                    nireq = exeNetwork.GetMetric(key).as<unsigned int>();
                } catch (const std::exception& ex) {
                    IE_THROW() << "Every device used with the benchmark_app should "
                               << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS "
                                  "ExecutableNetwork metric. "
                               << "Failed to query the metric for the " << device_name << " with error:" << ex.what();
                }
            }
        }

        // Iteration limit
        uint32_t niter = FLAGS_niter;
        if ((niter > 0) && (FLAGS_api == "async")) {
            niter = ((niter + nireq - 1) / nireq) * nireq;
            if (FLAGS_niter != niter) {
                slog::warn << "Number of iterations was aligned by request number from " << FLAGS_niter << " to " << niter << " using number of requests "
                           << nireq << slog::endl;
            }
        }

        // Time limit
        uint32_t duration_seconds = 0;
        if (FLAGS_t != 0) {
            // time limit
            duration_seconds = FLAGS_t;
        } else if (FLAGS_niter == 0) {
            // default time limit
            duration_seconds = deviceDefaultDeviceDurationInSeconds(device_name);
        }
        uint64_t duration_nanoseconds = getDurationInNanoseconds(duration_seconds);

        if (statistics) {
            statistics->addParameters(StatisticsReport::Category::RUNTIME_CONFIG,
                                      {
                                          {"topology", topology_name},
                                          {"target device", device_name},
                                          {"API", FLAGS_api},
                                          {"precision", std::string(precision.name())},
                                          {"batch size", std::to_string(batchSize)},
                                          {"number of iterations", std::to_string(niter)},
                                          {"number of parallel infer requests", std::to_string(nireq)},
                                          {"duration (ms)", std::to_string(getDurationInMilliseconds(duration_seconds))},
                                      });
            for (auto& nstreams : device_nstreams) {
                std::stringstream ss;
                ss << "number of " << nstreams.first << " streams";
                statistics->addParameters(StatisticsReport::Category::RUNTIME_CONFIG, {
                                                                                          {ss.str(), nstreams.second},
                                                                                      });
            }
        }

        // ----------------- 9. Creating infer requests and filling input blobs
        // ----------------------------------------
        next_step();

        InferRequestsQueue inferRequestsQueue(exeNetwork, nireq);
        fillBlobs(inputFiles, batchSize, app_inputs_info, inferRequestsQueue.requests);

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
            ss << getDurationInMilliseconds(duration_seconds) << " ms duration";
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

        // warming up - out of scope
        auto inferRequest = inferRequestsQueue.getIdleRequest();
        if (!inferRequest) {
            IE_THROW() << "No idle Infer Requests!";
        }
        if (FLAGS_api == "sync") {
            inferRequest->infer();
        } else {
            inferRequest->startAsync();
        }
        inferRequestsQueue.waitAll();
        auto duration_ms = double_to_string(inferRequestsQueue.getLatencies()[0]);
        slog::info << "First inference took " << duration_ms << " ms" << slog::endl;
        if (statistics)
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"first inference time (ms)", duration_ms}});
        inferRequestsQueue.resetTimes();

        auto startTime = Time::now();
        auto execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

        /** Start inference & calculate performance **/
        /** to align number if iterations to guarantee that last infer requests are
         * executed in the same conditions **/
        ProgressBar progressBar(progressBarTotalCount, FLAGS_stream_output, FLAGS_progress);

        while ((niter != 0LL && iteration < niter) || (duration_nanoseconds != 0LL && (uint64_t)execTime < duration_nanoseconds) ||
               (FLAGS_api == "async" && iteration % nireq != 0)) {
            inferRequest = inferRequestsQueue.getIdleRequest();
            if (!inferRequest) {
                IE_THROW() << "No idle Infer Requests!";
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
                inferRequest->startAsync();
            }
            iteration++;

            execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

            if (niter > 0) {
                progressBar.addProgress(1);
            } else {
                // calculate how many progress intervals are covered by current
                // iteration. depends on the current iteration time and time of each
                // progress interval. Previously covered progress intervals must be
                // skipped.
                auto progressIntervalTime = duration_nanoseconds / progressBarTotalCount;
                size_t newProgress = execTime / progressIntervalTime - progressCnt;
                progressBar.addProgress(newProgress);
                progressCnt += newProgress;
            }
        }

        // wait the latest inference executions
        inferRequestsQueue.waitAll();

        double latency = getMedianValue<double>(inferRequestsQueue.getLatencies());
        double totalDuration = inferRequestsQueue.getDurationInMilliseconds();
        double fps = (FLAGS_api == "sync") ? batchSize * 1000.0 / latency : batchSize * 1000.0 * iteration / totalDuration;

        if (statistics) {
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {
                                                                                         {"total execution time (ms)", double_to_string(totalDuration)},
                                                                                         {"total number of iterations", std::to_string(iteration)},
                                                                                     });
            if (device_name.find("MULTI") == std::string::npos) {
                statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {
                                                                                             {"latency (ms)", double_to_string(latency)},
                                                                                         });
            }
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {{"throughput", double_to_string(fps)}});
        }

        progressBar.finish();

        // ----------------- 11. Dumping statistics report
        // -------------------------------------------------------------
        next_step();

#ifdef USE_OPENCV
        if (!FLAGS_dump_config.empty()) {
            dump_config(FLAGS_dump_config, config);
            slog::info << "Inference Engine configuration settings were dumped to " << FLAGS_dump_config << slog::endl;
        }
#endif

        if (!FLAGS_exec_graph_path.empty()) {
            try {
                CNNNetwork execGraphInfo = exeNetwork.GetExecGraphInfo();
                execGraphInfo.serialize(FLAGS_exec_graph_path);
                slog::info << "executable graph is stored to " << FLAGS_exec_graph_path << slog::endl;
            } catch (const std::exception& ex) {
                slog::err << "Can't get executable graph: " << ex.what() << slog::endl;
            }
        }

        if (perf_counts) {
            std::vector<std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfCounts;
            for (size_t ireq = 0; ireq < nireq; ireq++) {
                auto reqPerfCounts = inferRequestsQueue.requests[ireq]->getPerformanceCounts();
                if (FLAGS_pc) {
                    slog::info << "Performance counts for " << ireq << "-th infer request:" << slog::endl;
                    printPerformanceCounts(reqPerfCounts, std::cout, getFullDeviceName(ie, FLAGS_d), false);
                }
                perfCounts.push_back(reqPerfCounts);
            }
            if (statistics) {
                statistics->dumpPerformanceCounters(perfCounts);
            }
        }

        if (statistics)
            statistics->dump();

        std::cout << "Count:      " << iteration << " iterations" << std::endl;
        std::cout << "Duration:   " << double_to_string(totalDuration) << " ms" << std::endl;
        if (device_name.find("MULTI") == std::string::npos)
            std::cout << "Latency:    " << double_to_string(latency) << " ms" << std::endl;
        std::cout << "Throughput: " << double_to_string(fps) << " FPS" << std::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;

        if (statistics) {
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS, {
                                                                                         {"error", ex.what()},
                                                                                     });
            statistics->dump();
        }

        return 3;
    }

    return 0;
}
