// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

std::shared_ptr<ov::Model> make_model(bool dynamic) {
    const ov::PartialShape input_shape = dynamic ? ov::PartialShape{-1, 32, -1, -1}
                                                  : ov::PartialShape{1, 32, 112, 112};
    auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32, input_shape);

    std::vector<float> weights1(64 * 32 * 3 * 3);
    std::vector<float> weights2(64 * 64 * 3 * 3);
    for (size_t i = 0; i < weights1.size(); ++i)
        weights1[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.01f;
    for (size_t i = 0; i < weights2.size(); ++i)
        weights2[i] = std::cos(static_cast<float>(i) * 0.007f) * 0.01f;

    auto w1 = ov::opset13::Constant::create(ov::element::f32, {64, 32, 3, 3}, weights1);
    auto w2 = ov::opset13::Constant::create(ov::element::f32, {64, 64, 3, 3}, weights2);
    auto value = std::make_shared<ov::opset13::Convolution>(input,
                                                            w1,
                                                            ov::Strides{1, 1},
                                                            ov::CoordinateDiff{1, 1},
                                                            ov::CoordinateDiff{1, 1},
                                                            ov::Strides{1, 1});
    for (size_t i = 0; i < 3; ++i) {
        value = std::make_shared<ov::opset13::Convolution>(
            std::make_shared<ov::opset13::Relu>(value),
            w2,
            ov::Strides{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::CoordinateDiff{1, 1},
            ov::Strides{1, 1});
    }
    return std::make_shared<ov::Model>(ov::OutputVector{value}, ov::ParameterVector{input}, "ocl_regression");
}

ov::Tensor make_input(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i)
        data[i] = std::sin(static_cast<float>(i) * 0.003f);
    return tensor;
}

double checksum(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    double result = 0.0;
    for (size_t i = 0; i < tensor.get_size(); ++i)
        result += static_cast<double>(data[i]);
    return result;
}

double infer_once(ov::InferRequest& request, const ov::Shape& shape, double* result = nullptr) {
    request.set_input_tensor(make_input(shape));
    const auto begin = Clock::now();
    request.infer();
    const double duration = elapsed_ms(begin);
    if (result)
        *result = checksum(request.get_output_tensor());
    return duration;
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
    return values.at(std::min(index, values.size() - 1));
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " DEVICE [ITERATIONS] [CACHE_DIR] [IR_PREFIX] [THROUGHPUT_BATCHES]\n";
        return 2;
    }

    const std::string device = argv[1];
    const size_t iterations = argc > 2 ? std::stoul(argv[2]) : 30;
    const std::string cache_dir = argc > 3 ? argv[3] : std::string{};
    const std::string ir_prefix = argc > 4 ? argv[4] : std::string{};
    const size_t throughput_batches = argc > 5 ? std::stoul(argv[5]) : 5;

    try {
        ov::Core core;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "device=" << device << '\n';

        if (device == "BOTH") {
            const auto available = core.get_available_devices();
            const auto has_device = [&](const std::string& name) {
                return std::find(available.begin(), available.end(), name) != available.end();
            };
            if (!has_device("GPU.0") || !has_device("GPU.1"))
                throw std::runtime_error("GPU.0 and GPU.1 must both be available");

            auto model = make_model(false);
            auto compiled0 = core.compile_model(model, "GPU.0");
            auto compiled1 = core.compile_model(model, "GPU.1");
            auto request0 = compiled0.create_infer_request();
            auto request1 = compiled1.create_infer_request();
            double checksum0 = 0.0;
            double checksum1 = 0.0;
            for (size_t i = 0; i < 5; ++i) {
                infer_once(request0, {1, 32, 112, 112}, &checksum0);
                infer_once(request1, {1, 32, 112, 112}, &checksum1);
            }
            request0.set_input_tensor(make_input({1, 32, 112, 112}));
            request1.set_input_tensor(make_input({1, 32, 112, 112}));
            request0.start_async();
            request1.start_async();
            request0.wait();
            request1.wait();
            if (!std::isfinite(checksum0) || !std::isfinite(checksum1))
                throw std::runtime_error("non-finite dual-device result");
            std::cout << "gpu0_checksum=" << std::setprecision(9) << checksum0 << '\n';
            std::cout << "gpu1_checksum=" << checksum1 << std::setprecision(3) << '\n';
            std::cout << "alternating_compile_infer=PASS\n";
            std::cout << "simultaneous_async=PASS\n";
            return 0;
        }

        std::cout << "full_name=" << core.get_property(device, ov::device::full_name) << '\n';
        std::cout << "device_id=" << core.get_property(device, ov::device::id) << '\n';
        std::cout << "device_type=" << core.get_property(device, ov::device::type) << '\n';

        auto model = make_model(false);
        if (!ir_prefix.empty())
            ov::serialize(model, ir_prefix + ".xml", ir_prefix + ".bin");

        auto begin = Clock::now();
        auto compiled = core.compile_model(model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        std::cout << "compile_ms=" << elapsed_ms(begin) << '\n';

        auto request = compiled.create_infer_request();
        double static_checksum = 0.0;
        std::cout << "first_infer_ms=" << infer_once(request, {1, 32, 112, 112}, &static_checksum) << '\n';
        std::vector<double> warm;
        warm.reserve(iterations);
        for (size_t i = 0; i < iterations; ++i)
            warm.push_back(infer_once(request, {1, 32, 112, 112}));
        std::cout << "warm_median_ms=" << percentile(warm, 0.50) << '\n';
        std::cout << "warm_p95_ms=" << percentile(warm, 0.95) << '\n';
        std::cout << "checksum=" << std::setprecision(9) << static_checksum << std::setprecision(3) << '\n';

        std::stringstream blob(std::ios::in | std::ios::out | std::ios::binary);
        compiled.export_model(blob);
        blob.seekg(0);
        begin = Clock::now();
        auto imported = core.import_model(blob, device);
        std::cout << "import_ms=" << elapsed_ms(begin) << '\n';
        auto imported_request = imported.create_infer_request();
        double imported_checksum = 0.0;
        std::cout << "imported_infer_ms="
                  << infer_once(imported_request, {1, 32, 112, 112}, &imported_checksum) << '\n';
        std::cout << "imported_checksum=" << std::setprecision(9) << imported_checksum << std::setprecision(3) << '\n';

        auto dynamic_model = make_model(true);
        begin = Clock::now();
        auto dynamic_compiled = core.compile_model(dynamic_model, device);
        std::cout << "dynamic_compile_ms=" << elapsed_ms(begin) << '\n';
        auto dynamic_request = dynamic_compiled.create_infer_request();
        double dynamic_checksum_a = 0.0;
        double dynamic_checksum_b = 0.0;
        std::cout << "dynamic_a_ms=" << infer_once(dynamic_request, {1, 32, 64, 64}, &dynamic_checksum_a) << '\n';
        std::cout << "dynamic_b_ms=" << infer_once(dynamic_request, {2, 32, 96, 80}, &dynamic_checksum_b) << '\n';
        std::cout << "dynamic_checksums=" << std::setprecision(9) << dynamic_checksum_a << ',' << dynamic_checksum_b
                  << std::setprecision(3) << '\n';

        auto throughput_model = core.compile_model(
            model,
            device,
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        std::vector<ov::InferRequest> requests;
        for (size_t i = 0; i < 4; ++i) {
            requests.push_back(throughput_model.create_infer_request());
            requests.back().set_input_tensor(make_input({1, 32, 112, 112}));
        }
        begin = Clock::now();
        for (size_t batch = 0; batch < throughput_batches; ++batch) {
            for (auto& current : requests)
                current.start_async();
            for (auto& current : requests)
                current.wait();
        }
        const double concurrent_ms = elapsed_ms(begin);
        const size_t concurrent_inferences = throughput_batches * requests.size();
        std::cout << "concurrent_inferences=" << concurrent_inferences << '\n';
        std::cout << "concurrent_ms=" << concurrent_ms << '\n';
        std::cout << "concurrent_fps=" << 1000.0 * concurrent_inferences / concurrent_ms << '\n';

        if (!cache_dir.empty()) {
            std::filesystem::create_directories(cache_dir);
            ov::Core cache_core;
            cache_core.set_property(ov::cache_dir(cache_dir));
            begin = Clock::now();
            auto cache_first = cache_core.compile_model(model, device);
            std::cout << "cache_first_compile_ms=" << elapsed_ms(begin) << '\n';
            begin = Clock::now();
            auto cache_second = cache_core.compile_model(model, device);
            std::cout << "cache_second_compile_ms=" << elapsed_ms(begin) << '\n';
            auto cache_request = cache_second.create_infer_request();
            double cache_checksum = 0.0;
            std::cout << "cache_infer_ms=" << infer_once(cache_request, {1, 32, 112, 112}, &cache_checksum) << '\n';
            std::cout << "cache_checksum=" << std::setprecision(9) << cache_checksum << std::setprecision(3) << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
