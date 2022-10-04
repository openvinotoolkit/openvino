// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

// TODO: update tests
// TODO: add readme
// TODO: add comments
int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;
        if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model>" << slog::endl;
            return EXIT_FAILURE;
        }
        ov::Core core;
        // Set nstreams to 1 to make synchronous inference. For CPU its default value is already 1,
        // but other devises like MYRIAD may have a different default value
        ov::AnyMap nstreams{{ov::streams::num.name(), 1}};
        // Replace CPU with your device, for example AUTO
        ov::CompiledModel compiled_model = core.compile_model(argv[1], "CPU", nstreams);
        ov::InferRequest ireq = compiled_model.create_infer_request();
        for (const ov::Output<const ov::Node>& input : compiled_model.inputs()) {
            fill_tensor_random(ireq.get_tensor(input));
        }

        // Warm up
        ireq.infer();

        int niter = 100;
        std::vector<double> latencies;
        latencies.reserve(niter);
        auto start = std::chrono::steady_clock::now();
        auto time_point = start;
        for (int i = 0; i < niter; ++i) {
            ireq.infer();
            auto iter_end = std::chrono::steady_clock::now();
            latencies.push_back(std::chrono::duration_cast<Ms>(iter_end - time_point).count());
            time_point = iter_end;
        }
        auto end = time_point;

        auto duration = std::chrono::duration_cast<Ms>(end - start).count();
        slog::info << "Count:      " << niter << " iterations" << slog::endl;
        slog::info << "Duration:   " << duration << " ms" << slog::endl;
        slog::info << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics latency_metrics{latencies, "", percent};
        latency_metrics.write_to_slog();
        slog::info << "Throughput: " << double_to_string(niter * 1000 / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 1;
    }
    return 0;
}
