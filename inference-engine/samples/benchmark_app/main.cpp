// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "progress_bar.hpp"
#include "statistics_report.hpp"
#include "inputs_filling.hpp"
#include "utils.hpp"

using namespace InferenceEngine;

static const size_t progressBarDefaultTotalCount = 1000;

uint64_t getDurationInMilliseconds(uint32_t duration) {
    return duration * 1000LL;
}

uint64_t getDurationInNanoseconds(uint32_t duration) {
    return duration * 1000000000LL;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_api != "async" && FLAGS_api != "sync") {
        throw std::logic_error("Incorrect API. Please set -api option to `sync` or `async` value.");
    }

    if (!FLAGS_report_type.empty() &&
         FLAGS_report_type != noCntReport && FLAGS_report_type != averageCntReport && FLAGS_report_type != detailedCntReport) {
        std::string err = "only " + std::string(noCntReport) + "/" + std::string(averageCntReport) + "/" + std::string(detailedCntReport) +
                " report types are supported (invalid -report_type option value)";
        throw std::logic_error(err);
    }

    if ((FLAGS_report_type == averageCntReport) && ((FLAGS_d.find("MULTI") != std::string::npos))) {
        throw std::logic_error("only " + std::string(detailedCntReport) + " report type is supported for MULTI device");
    }

    return true;
}

static void next_step(const std::string additional_info = "") {
    static size_t step_id = 0;
    static const std::map<size_t, std::string> step_names = {
      { 1, "Parsing and validating input arguments" },
      { 2, "Loading Inference Engine" },
      { 3, "Reading the Intermediate Representation network" },
      { 4, "Resizing network to match image sizes and given batch" },
      { 5, "Configuring input of the model" },
      { 6, "Setting device configuration" },
      { 7, "Loading the model to the device" },
      { 8, "Setting optimal runtime parameters" },
      { 9, "Creating infer requests and filling input blobs with images" },
      { 10, "Measuring performance" },
      { 11, "Dumping statistics report" }
    };

    step_id++;
    if (step_names.count(step_id) == 0)
        THROW_IE_EXCEPTION << "Step ID " << step_id << " is out of total steps number " << step_names.size();

    std::cout << "[Step " << step_id << "/" << step_names.size() << "] " << step_names.at(step_id)
              << (additional_info.empty() ? "" : " (" + additional_info + ")") << std::endl;
}

template <typename T>
T getMedianValue(const std::vector<T> &vec) {
    std::vector<T> sortedVec(vec);
    std::sort(sortedVec.begin(), sortedVec.end());
    return (sortedVec.size() % 2 != 0) ?
           sortedVec[sortedVec.size() / 2ULL] :
           (sortedVec[sortedVec.size() / 2ULL] + sortedVec[sortedVec.size() / 2ULL - 1ULL]) / static_cast<T>(2.0);
}

/**
* @brief The entry point of the benchmark application
*/
int main(int argc, char *argv[]) {
    std::shared_ptr<StatisticsReport> statistics;
    try {
        // ----------------- 1. Parsing and validating input arguments -------------------------------------------------
        next_step();

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        if (!FLAGS_report_type.empty()) {
            std::vector<gflags::CommandLineFlagInfo> flags;
            StatisticsReport::Parameters command_line_arguments;
            gflags::GetAllFlags(&flags);

            for (auto &flag : flags) {
                if (!flag.is_default) {
                    command_line_arguments.push_back({ flag.name, flag.current_value });
                }
            }
            statistics = std::make_shared<StatisticsReport>(StatisticsReport::Config{FLAGS_report_type, FLAGS_report_folder});
            statistics->addParameters(StatisticsReport::Category::COMMAND_LINE_PARAMETERS, command_line_arguments);
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> inputFiles;
        parseInputFilesArguments(inputFiles);

        if (FLAGS_nstreams.empty()) {
            slog::warn << "-nstreams default value is determined automatically for a device. "
                "Although the automatic selection usually provides a reasonable performance,"
                "but it still may be non-optimal for some cases, for more information look at README." << slog::endl<< slog::endl;
        }

        // ----------------- 2. Loading the Inference Engine -----------------------------------------------------------
        next_step();

        // Get optimal runtime parameters for device
        std::string device_name = FLAGS_d;

        Core ie;

        if (FLAGS_d.find("CPU") != std::string::npos) {
            // Loading default CPU extensions
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

            if (!FLAGS_l.empty()) {
                // CPU (MKLDNN) extensions is loaded as a shared library and passed as a pointer to base extension
                const auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
                ie.AddExtension(extension_ptr, "CPU");
                slog::info << "CPU (MKLDNN) extensions is loaded " << FLAGS_l << slog::endl;
            }
        }

        if ((FLAGS_d.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
            // Load clDNN Extensions
            ie.SetConfig({ {CONFIG_KEY(CONFIG_FILE), FLAGS_c} });
            slog::info << "GPU extensions is loaded " << FLAGS_c << slog::endl;
        }

        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(device_name) << std::endl;

        // ----------------- 3. Reading the Intermediate Representation network ----------------------------------------
        next_step();

        slog::info << "Loading network files" << slog::endl;

        CNNNetReader netBuilder;
        auto startTime = Time::now();
        netBuilder.ReadNetwork(FLAGS_m);
        const std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netBuilder.ReadWeights(binFileName);
        auto float_to_string = [] (const float number) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << number;
            return ss.str();
        };
        auto get_total_ms_time = [ &startTime ] () {
            return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
        };
        auto duration_ms = float_to_string(get_total_ms_time());
        slog::info << "Read network took " << duration_ms << " ms" << slog::endl;
        if (statistics)
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                      {
                                          {"read network time (ms)", duration_ms}
                                      });

        CNNNetwork cnnNetwork = netBuilder.getNetwork();
        const InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.empty()) {
            throw std::logic_error("no inputs info is provided");
        }

        // ----------------- 4. Resizing network to match image sizes and given batch ----------------------------------
        next_step();

        if (FLAGS_b != 0) {
            ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
            bool reshape = false;
            for (const InputsDataMap::value_type& item : inputInfo) {
                auto layout = item.second->getTensorDesc().getLayout();

                int batchIndex = -1;
                if ((layout == Layout::NCHW) || (layout == Layout::NCDHW) ||
                    (layout == Layout::NHWC) || (layout == Layout::NDHWC) ||
                    (layout == Layout::NC)) {
                    batchIndex = 0;
                } else if (layout == CN) {
                    batchIndex = 1;
                }
                if ((batchIndex != -1) && (shapes[item.first][batchIndex] != FLAGS_b)) {
                    shapes[item.first][batchIndex] = FLAGS_b;
                    reshape = true;
                }
            }
            if (reshape) {
                slog::info << "Resizing network to batch = " << FLAGS_b << slog::endl;
                cnnNetwork.reshape(shapes);
            }
        }

        const size_t batchSize = cnnNetwork.getBatchSize();
        const Precision precision = cnnNetwork.getPrecision();
        slog::info << (FLAGS_b != 0 ? "Network batch size was changed to: " : "Network batch size: ") << batchSize <<
            ", precision: " << precision << slog::endl;

        // ----------------- 5. Configuring input ----------------------------------------------------------------------
        next_step();

        for (auto& item : inputInfo) {
            if (isImage(item.second)) {
                /** Set the precision of input data provided by the user, should be called before load of the network to the device **/
                item.second->setPrecision(Precision::U8);
            }
        }

        // ----------------- 6. Setting device configuration -----------------------------------------------------------
        next_step();

        bool perf_counts = (FLAGS_report_type == detailedCntReport ||
                            FLAGS_report_type == averageCntReport ||
                            FLAGS_pc ||
                            !FLAGS_exec_graph_path.empty());

        auto devices = parseDevices(device_name);
        std::map<std::string, uint32_t> device_nstreams = parseValuePerDevice(devices, FLAGS_nstreams);
        for (auto& device : devices) {
            if (device == "CPU") {  // CPU supports few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (FLAGS_nthreads != 0)
                    ie.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads) }}, device);

                if ((device_name.find("MULTI") != std::string::npos) &&
                    (device_name.find("GPU") != std::string::npos)) {
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, device);
                } else {
                    // pin threads for CPU portion of inference
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), FLAGS_pin }}, device);
                }

                // for CPU execution, more throughput-oriented execution via streams
                if (FLAGS_api == "async")
                    ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                    (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                         "CPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            } else if (device == ("GPU")) {
                if (FLAGS_api == "async")
                    ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                    (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                         "GPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

                if ((device_name.find("MULTI") != std::string::npos) &&
                    (device_name.find("CPU") != std::string::npos)) {
                    // multi-device execution with the CPU + GPU performs best with GPU trottling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    ie.SetConfig({{ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" }}, "GPU");
                }
            } else if (device == "MYRIAD") {
                ie.SetConfig({{ CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
                              { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) }}, device);
            }
        }

        // ----------------- 7. Loading the model to the device --------------------------------------------------------
        next_step();

        std::map<std::string, std::string> config = {{ CONFIG_KEY(PERF_COUNT), perf_counts ? CONFIG_VALUE(YES) :
                                                                                             CONFIG_VALUE(NO) }};
        startTime = Time::now();
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device_name, config);
        duration_ms = float_to_string(get_total_ms_time());
        slog::info << "Load network took " << duration_ms << " ms" << slog::endl;
        if (statistics)
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                      {
                                          {"load network time (ms)", duration_ms}
                                      });

        // ----------------- 8. Setting optimal runtime parameters -----------------------------------------------------
        next_step();

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if (nireq == 0) {
            std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            try {
                nireq = exeNetwork.GetMetric(key).as<unsigned int>();
            } catch (const details::InferenceEngineException& ex) {
                THROW_IE_EXCEPTION
                        << "Every device used with the benchmark_app should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << device_name << " with error:" << ex.what();
            }
        }

        // Iteration limit
        uint32_t niter = FLAGS_niter;
        if ((niter > 0) && (FLAGS_api == "async")) {
            niter = ((niter + nireq - 1)/nireq)*nireq;
            if (FLAGS_niter != niter) {
                slog::warn << "Number of iterations was aligned by request number from "
                           << FLAGS_niter << " to " << niter << " using number of requests " << nireq << slog::endl;
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
                                            {"topology", cnnNetwork.getName()},
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
                statistics->addParameters(StatisticsReport::Category::RUNTIME_CONFIG,
                                          {
                                                {ss.str(), std::to_string(nstreams.second)},
                                          });
            }
        }

        // ----------------- 9. Creating infer requests and filling input blobs ----------------------------------------
        next_step();

        InferRequestsQueue inferRequestsQueue(exeNetwork, nireq);

        fillBlobs(inputFiles, batchSize, inputInfo, inferRequestsQueue.requests);

        // ----------------- 10. Measuring performance ------------------------------------------------------------------
        size_t progressCnt = 0;
        size_t progressBarTotalCount = progressBarDefaultTotalCount;
        size_t iteration = 0;

        std::stringstream ss;
        ss << "Start inference " << FLAGS_api << "ronously";
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
            THROW_IE_EXCEPTION << "No idle Infer Requests!";
        }

        if (FLAGS_api == "sync") {
            inferRequest->infer();
        } else {
            inferRequest->startAsync();
        }
        inferRequestsQueue.waitAll();
        inferRequestsQueue.resetTimes();

        startTime = Time::now();
        auto execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

        /** Start inference & calculate performance **/
        /** to align number if iterations to guarantee that last infer requests are executed in the same conditions **/
        ProgressBar progressBar(progressBarTotalCount, FLAGS_stream_output, FLAGS_progress);

        while ((niter != 0LL && iteration < niter) ||
               (duration_nanoseconds != 0LL && (uint64_t)execTime < duration_nanoseconds) ||
               (FLAGS_api == "async" && iteration % nireq != 0)) {
            inferRequest = inferRequestsQueue.getIdleRequest();
            if (!inferRequest) {
                THROW_IE_EXCEPTION << "No idle Infer Requests!";
            }

            if (FLAGS_api == "sync") {
                inferRequest->infer();
            } else {
                inferRequest->startAsync();
            }
            iteration++;

            execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

            if (niter > 0) {
                progressBar.addProgress(1);
            } else {
                // calculate how many progress intervals are covered by current iteration.
                // depends on the current iteration time and time of each progress interval.
                // Previously covered progress intervals must be skipped.
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
        double fps = (FLAGS_api == "sync") ? batchSize * 1000.0 / latency :
                                             batchSize * 1000.0 * iteration / totalDuration;

        if (statistics) {
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                      {
                                        {"total execution time (ms)", float_to_string(totalDuration)},
                                        {"total number of iterations", std::to_string(iteration)},
                                      });
            if (device_name.find("MULTI") == std::string::npos) {
                statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                          {
                                            {"latency (ms)", float_to_string(latency)},
                                          });
            }
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                      {
                                          {"throughput", float_to_string(fps)}
                                      });
        }

        progressBar.finish();

        // ----------------- 11. Dumping statistics report -------------------------------------------------------------
        next_step();

        if (!FLAGS_exec_graph_path.empty()) {
            try {
                CNNNetwork execGraphInfo = exeNetwork.GetExecGraphInfo();
                execGraphInfo.serialize(FLAGS_exec_graph_path);
                slog::info << "executable graph is stored to " << FLAGS_exec_graph_path << slog::endl;
            } catch (const std::exception & ex) {
                slog::err << "Can't get executable graph: " << ex.what() << slog::endl;
            }
        }

        if (perf_counts) {
            std::vector<std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfCounts;
            for (size_t ireq = 0; ireq < nireq; ireq++) {
                auto reqPerfCounts = inferRequestsQueue.requests[ireq]->getPerformanceCounts();
                if (FLAGS_pc) {
                    slog::info << "Pefrormance counts for " << ireq << "-th infer request:" << slog::endl;
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
        std::cout << "Duration:   " << float_to_string(totalDuration) << " ms" << std::endl;
        if (device_name.find("MULTI") == std::string::npos)
            std::cout << "Latency:    " << float_to_string(latency) << " ms" << std::endl;
        std::cout << "Throughput: " << float_to_string(fps) << " FPS" << std::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;

        if (statistics) {
            statistics->addParameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                      {
                                            {"error", ex.what()},
                                      });
            statistics->dump();
        }

        return 3;
    }

    return 0;
}
