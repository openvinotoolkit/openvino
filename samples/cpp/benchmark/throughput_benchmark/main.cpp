// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <condition_variable>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
// clang-format on

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;
        if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model>" << slog::endl;
            return EXIT_FAILURE;
        }
        // Optimize for throughput. Best throughput can be reached by
        // running multiple ov::InferRequest instances asyncronously
        ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};

        // Create ov::Core and use it to compile a model
        // Pick device by replacing CPU, for example MULTI:CPU(4),GPU(8)
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(argv[1], "CPU", tput);
        // Create optimal number of ov::InferRequest instances
        uint32_t nireq;
        try {
            nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (const std::exception& ex) {
            throw std::runtime_error("Every used device must support " +
                                     std::string(ov::optimal_number_of_infer_requests.name()) +
                                     " Failed to query the property with error: " + ex.what());
        }
        std::vector<ov::InferRequest> ireqs;
        for (uint32_t i = 0; i < nireq; ++i) {
            ireqs.push_back(compiled_model.create_infer_request());
        }
        // Fill input data for ireqs
        for (ov::InferRequest& ireq : ireqs) {
            for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
                fill_tensor_random(ireq.get_tensor(model_input));
            }
        }

        // Warm up
        for (ov::InferRequest& ireq : ireqs) {
            ireq.start_async();
        }
        for (ov::InferRequest& ireq : ireqs) {
            ireq.wait();
        }
        // Run benchmarking for seconds_to_run seconds
        std::chrono::seconds seconds_to_run{15};
        std::vector<double> latencies;
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<std::chrono::steady_clock::time_point> time_points(nireq);
        std::exception_ptr callback_exception;
        int nstarted = 0;
        auto start = std::chrono::steady_clock::now();
        auto time_point_to_finish = start + seconds_to_run;
        // For each ireq set up a callback which does the actual management of results and asynchronous infer.
        // When inference finishes, the callback updates latencies vector and starts new inference.
        // After the main loop wait for all ireqs completion
        for (uint32_t i = 0; i < nireq; ++i) {
            ov::InferRequest& ireq = ireqs[i];
            std::chrono::steady_clock::time_point& time_point = time_points[i];
            time_point = std::chrono::steady_clock::now();
            ireq.set_callback(
                [&ireq, &time_point, &mutex, &cv, &latencies, &callback_exception, &nstarted, time_point_to_finish](
                    std::exception_ptr ex) {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        try {
                            if (ex) {
                                std::rethrow_exception(ex);
                            }
                            auto infer_end = std::chrono::steady_clock::now();
                            latencies.push_back(std::chrono::duration_cast<Ms>(infer_end - time_point).count());
                            if (latencies.size() >= nstarted) {
                                cv.notify_one();
                                return;
                            }
                            if (infer_end < time_point_to_finish) {
                                time_point = infer_end;
                                ++nstarted;
                                ireq.start_async();
                            }
                        } catch (const std::exception&) {
                            if (!callback_exception) {
                                callback_exception = std::current_exception();
                                cv.notify_one();
                            }
                        }
                    }
                });
            std::unique_lock<std::mutex> lock(mutex);
            ireq.start_async();
            ++nstarted;
        }
        std::unique_lock<std::mutex> lock(mutex);
        while (!callback_exception && latencies.size() < nstarted) {
            cv.wait(lock);
        }
        if (callback_exception) {
            std::rethrow_exception(callback_exception);
        }
        auto end = std::chrono::steady_clock::now();
        // Report results
        double duration = std::chrono::duration_cast<Ms>(end - start).count();
        slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl;
        slog::info << "Duration:   " << duration << " ms" << slog::endl;
        slog::info << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics latency_metrics{latencies, "", percent};
        latency_metrics.write_to_slog();
        slog::info << "Throughput: " << double_to_string(latencies.size() * 1000 / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
