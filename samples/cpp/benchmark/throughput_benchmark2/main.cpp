// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <condition_variable>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
// clang-format on

namespace {
using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fill_random(ov::Tensor& tensor,
                 T rand_min = std::numeric_limits<uint8_t>::min(),
                 T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);  // TODO: why is gen always reset?
    size_t tensor_size = tensor.get_size();
    if (0 == tensor_size) {
        throw std::runtime_error("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    }
    T* data = tensor.data<T>();
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = static_cast<T>(distribution(gen));
    }
}

void fill_tensor_random(ov::Tensor tensor) {
    switch (tensor.get_element_type()) {
        case ov::element::f32:
            fill_random<float, float>(tensor);
            break;
        case ov::element::f64:
            fill_random<double, double>(tensor);
            break;
        case ov::element::f16:
            fill_random<short, short>(tensor);
            break;
        case ov::element::i32:
            fill_random<int32_t, int32_t>(tensor);
            break;
        case ov::element::i64:
            fill_random<int64_t, int64_t>(tensor);
            break;
        case ov::element::u8:
            // uniform_int_distribution<uint8_t> is not allowed in the C++17
            // standard and vs2017/19
            fill_random<uint8_t, uint32_t>(tensor);
            break;
        case ov::element::i8:
            // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
            // and vs2017/19
            fill_random<int8_t, int32_t>(tensor,
                                         std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max());
            break;
        case ov::element::u16:
            fill_random<uint16_t, uint16_t>(tensor);
            break;
        case ov::element::i16:
            fill_random<int16_t, int16_t>(tensor);
            break;
        case ov::element::boolean:
            fill_random<uint8_t, uint32_t>(tensor, 0, 1);
            break;
        default:
            throw ov::Exception("Input type is not supported for a tensor");
    }
}
}

int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;
        if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model>" << slog::endl;
            return EXIT_FAILURE;
        }
        ov::Core core;
        // TODO: explain throughput
        ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};

        // Pick device by replacing CPU, for example MULTI:CPU,GPU
        ov::CompiledModel compiled_model = core.compile_model(argv[1], "CPU", tput);
        uint32_t nireq;
        try {
            nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
        } catch (const std::exception& ex) {
            throw std::runtime_error("Every used device must support " +
                                     std::string(ov::optimal_number_of_infer_requests.name()) +
                                     " Failed to query the property with error:" + ex.what());
        }
        std::vector<ov::InferRequest> ireqs;
        for (uint32_t i = 0; i < nireq; ++i) {
            ireqs.push_back(compiled_model.create_infer_request());
        }
        for (ov::InferRequest& ireq : ireqs) {
            for (const ov::Output<const ov::Node>& input : compiled_model.inputs()) {
                fill_tensor_random(ireq.get_tensor(input));
            }
        }

        // Warm up
        for (ov::InferRequest& ireq : ireqs) {
            ireq.start_async();
        }
        for (ov::InferRequest& ireq : ireqs) {
            ireq.wait();
        }

        int init_niter = 1000;
        // Align number of iterations by request number
        int niter = ((init_niter + nireq - 1) / nireq) * nireq;
        if (init_niter != niter) {
            slog::warn << "Number of iterations was aligned by request number from " << init_niter << " to "
                        << niter << " using number of requests " << nireq << slog::endl;
        }
        std::vector<double> latencies;
        latencies.reserve(niter);
        std::mutex mutex;
        std::condition_variable cv;
        std::exception_ptr callback_exception;
        struct TimedIreq {
            ov::InferRequest& ireq;
            std::chrono::steady_clock::time_point start;
            bool has_start_time;
        };
        std::deque<TimedIreq> finished_ireqs;
        for (ov::InferRequest& ireq : ireqs) {
            finished_ireqs.push_back({ireq, std::chrono::steady_clock::time_point{}, false});
        }
        auto start = std::chrono::steady_clock::now();
        int iteration = 0;
        while (latencies.size() < niter) {
            std::unique_lock<std::mutex> lock(mutex);
            while (!callback_exception && finished_ireqs.empty()) {
                cv.wait(lock);
            }
            if (callback_exception) {
                std::rethrow_exception(callback_exception);
            }
            if (!finished_ireqs.empty()) {
                TimedIreq timedIreq = finished_ireqs.front();
                finished_ireqs.pop_front();
                lock.unlock();
                ov::InferRequest& ireq = timedIreq.ireq;
                auto time_point = std::chrono::steady_clock::now();
                if (timedIreq.has_start_time) {
                    latencies.push_back(std::chrono::duration_cast<Ms>(time_point - timedIreq.start).count());
                }
                ireq.set_callback(
                        [&ireq, time_point, &mutex, &finished_ireqs, &callback_exception, &cv] (std::exception_ptr ex) {
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
                if (iteration < niter) {
                    ireq.start_async();
                    ++iteration;
                }
            }
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<Ms>(end - start).count();
        slog::info << "Count:      " << niter << " iterations" << slog::endl;
        slog::info << "Duration:   " << duration << " ms" << slog::endl;
        slog::info << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics latency_metrics{latencies, "", percent};
        latency_metrics.write_to_slog();
        std::cout << duration << '\n';
        slog::info << "Throughput: " << double_to_string(1000 * niter / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 1;
    }
    return 0;
}
