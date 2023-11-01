// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
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
        slog::info << "OpenVINO:" << slog::endl;
        slog::info << ov::get_openvino_version();

        std::string device_name = "CPU";
        if (argc == 3) {
            device_name = TSTRING2STRING(argv[2]);
        } else if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <device_name>(default: CPU)" << slog::endl;
            return EXIT_FAILURE;
        }
        // Optimize for throughput. Best throughput can be reached by
        // running multiple ov::InferRequest instances asyncronously
        ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};

        // Create ov::Core and use it to compile a model.
        // Pick a device by replacing CPU, for example MULTI:CPU(4),GPU(8).
        // It is possible to set CUMULATIVE_THROUGHPUT as ov::hint::PerformanceMode for AUTO device
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(argv[1], device_name, tput);
        // Create optimal number of ov::InferRequest instances
        uint32_t nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
        std::vector<ov::InferRequest> ireqs(nireq);
        std::generate(ireqs.begin(), ireqs.end(), [&] {
            return compiled_model.create_infer_request();
        });
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
        // Benchmark for seconds_to_run seconds and at least niter iterations
        std::chrono::seconds seconds_to_run{10};
        size_t niter = 10;
        std::vector<double> latencies;
        std::mutex mutex;
        std::condition_variable cv;
        std::exception_ptr callback_exception;
        struct TimedIreq {
            ov::InferRequest& ireq;  // ref
            std::chrono::steady_clock::time_point start;
            bool has_start_time;
        };
        std::deque<TimedIreq> finished_ireqs;
        for (ov::InferRequest& ireq : ireqs) {
            finished_ireqs.push_back({ireq, std::chrono::steady_clock::time_point{}, false});
        }
        auto start = std::chrono::steady_clock::now();
        auto time_point_to_finish = start + seconds_to_run;
        // Once there’s a finished ireq wake up main thread.
        // Compute and save latency for that ireq and prepare for next inference by setting up callback.
        // Callback pushes that ireq again to finished ireqs when infrence is completed.
        // Start asynchronous infer with updated callback
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            while (!callback_exception && finished_ireqs.empty()) {
                cv.wait(lock);
            }
            if (callback_exception) {
                std::rethrow_exception(callback_exception);
            }
            if (!finished_ireqs.empty()) {
                auto time_point = std::chrono::steady_clock::now();
                if (time_point > time_point_to_finish && latencies.size() > niter) {
                    break;
                }
                TimedIreq timedIreq = finished_ireqs.front();
                finished_ireqs.pop_front();
                lock.unlock();
                ov::InferRequest& ireq = timedIreq.ireq;
                if (timedIreq.has_start_time) {
                    latencies.push_back(std::chrono::duration_cast<Ms>(time_point - timedIreq.start).count());
                }
                ireq.set_callback(
                    [&ireq, time_point, &mutex, &finished_ireqs, &callback_exception, &cv](std::exception_ptr ex) {
                        // Keep callback small. This improves performance for fast (tens of thousands FPS) models
                        std::unique_lock<std::mutex> lock(mutex);
                        {
                            try {
                                if (ex) {
                                    std::rethrow_exception(ex);
                                }
                                finished_ireqs.push_back({ireq, time_point, true});
                            } catch (const std::exception&) {
                                if (!callback_exception) {
                                    callback_exception = std::current_exception();
                                }
                            }
                        }
                        cv.notify_one();
                    });
                ireq.start_async();
            }
        }
        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<Ms>(end - start).count();
        // Report results
        slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl
                   << "Duration:   " << duration << " ms" << slog::endl
                   << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics{latencies, "", percent}.write_to_slog();
        slog::info << "Throughput: " << double_to_string(1000 * latencies.size() / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
