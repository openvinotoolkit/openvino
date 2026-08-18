// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

std::shared_ptr<ov::Model> make_model(bool dynamic) {
    const ov::PartialShape shape = dynamic ? ov::PartialShape{-1, -1} : ov::PartialShape{1024, 1024};
    auto lhs = std::make_shared<ov::opset13::Parameter>(ov::element::f32, shape);
    auto rhs = std::make_shared<ov::opset13::Parameter>(ov::element::f32, shape);
    ov::Output<ov::Node> value = std::make_shared<ov::opset13::Add>(lhs, rhs);
    for (size_t i = 0; i < 8; ++i) {
        value = std::make_shared<ov::opset13::Multiply>(value, rhs);
        value = std::make_shared<ov::opset13::Add>(value, lhs);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{value}, ov::ParameterVector{lhs, rhs}, "vulkan_eltwise");
}

ov::Tensor make_input(const ov::Shape& shape, float phase) {
    ov::Tensor tensor(ov::element::f32, shape);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i)
        data[i] = std::sin(static_cast<float>(i) * 0.003f + phase) * 0.1f;
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
    request.set_input_tensor(0, make_input(shape, 0.2f));
    request.set_input_tensor(1, make_input(shape, 1.1f));
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

bool matches(double lhs, double rhs) {
    return std::abs(lhs - rhs) <= 1e-5 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " DEVICE [ITERATIONS] [CACHE_DIR] [THROUGHPUT_BATCHES] [DYNAMIC_ITERATIONS]\n";
        return 2;
    }

    const std::string device = argv[1];
    const size_t iterations = argc > 2 ? std::stoul(argv[2]) : 30;
    const std::string cache_dir = argc > 3 ? argv[3] : std::string{};
    const size_t throughput_batches = argc > 4 ? std::stoul(argv[4]) : 5;
    const size_t dynamic_iterations = argc > 5 ? std::stoul(argv[5]) : 0;

    try {
        ov::Core core;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "device=" << device << '\n';
        std::cout << "full_name=" << core.get_property(device, ov::device::full_name) << '\n';
        std::cout << "device_id=" << core.get_property(device, ov::device::id) << '\n';

        auto model = make_model(false);
        auto begin = Clock::now();
        auto compiled = core.compile_model(model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        std::cout << "compile_ms=" << elapsed_ms(begin) << '\n';

        auto request = compiled.create_infer_request();
        double static_checksum = 0.0;
        std::cout << "first_infer_ms=" << infer_once(request, {1024, 1024}, &static_checksum) << '\n';
        std::vector<double> warm;
        warm.reserve(iterations);
        for (size_t i = 0; i < iterations; ++i)
            warm.push_back(infer_once(request, {1024, 1024}));
        std::cout << "warm_median_ms=" << percentile(warm, 0.50) << '\n';
        std::cout << "warm_p95_ms=" << percentile(warm, 0.95) << '\n';
        std::cout << "checksum=" << std::setprecision(9) << static_checksum << std::setprecision(3) << '\n';

        try {
            std::stringstream blob(std::ios::in | std::ios::out | std::ios::binary);
            compiled.export_model(blob);
            blob.seekg(0);
            begin = Clock::now();
            auto imported = core.import_model(blob, device);
            std::cout << "import_ms=" << elapsed_ms(begin) << '\n';
            auto imported_request = imported.create_infer_request();
            double imported_checksum = 0.0;
            std::cout << "imported_infer_ms="
                      << infer_once(imported_request, {1024, 1024}, &imported_checksum) << '\n';
            std::cout << "imported_checksum=" << std::setprecision(9) << imported_checksum << std::setprecision(3)
                      << '\n';
            std::cout << "export_import=" << (matches(static_checksum, imported_checksum) ? "PASS" : "FAIL") << '\n';
        } catch (const std::exception& error) {
            std::cout << "export_import=UNSUPPORTED:" << error.what() << '\n';
        }

        try {
            auto dynamic_model = make_model(true);
            begin = Clock::now();
            auto dynamic_compiled = core.compile_model(dynamic_model, device);
            std::cout << "dynamic_compile_ms=" << elapsed_ms(begin) << '\n';
            auto dynamic_request = dynamic_compiled.create_infer_request();
            double dynamic_checksum_a = 0.0;
            double dynamic_checksum_b = 0.0;
            std::cout << "dynamic_a_ms=" << infer_once(dynamic_request, {512, 512}, &dynamic_checksum_a) << '\n';
            std::cout << "dynamic_b_ms=" << infer_once(dynamic_request, {257, 513}, &dynamic_checksum_b) << '\n';
            std::cout << "dynamic_checksums=" << std::setprecision(9) << dynamic_checksum_a << ','
                      << dynamic_checksum_b << std::setprecision(3) << '\n';

            if (dynamic_iterations > 0) {
                std::vector<double> dynamic_a_warm;
                std::vector<double> dynamic_b_warm;
                dynamic_a_warm.reserve(dynamic_iterations);
                dynamic_b_warm.reserve(dynamic_iterations);
                for (size_t i = 0; i < dynamic_iterations; ++i) {
                    dynamic_a_warm.push_back(infer_once(dynamic_request, {1024, 1024}, &dynamic_checksum_a));
                    dynamic_b_warm.push_back(infer_once(dynamic_request, {512, 768}, &dynamic_checksum_b));
                }
                std::cout << "dynamic_a_warm_median_ms=" << percentile(dynamic_a_warm, 0.50) << '\n';
                std::cout << "dynamic_a_warm_p95_ms=" << percentile(dynamic_a_warm, 0.95) << '\n';
                std::cout << "dynamic_b_warm_median_ms=" << percentile(dynamic_b_warm, 0.50) << '\n';
                std::cout << "dynamic_b_warm_p95_ms=" << percentile(dynamic_b_warm, 0.95) << '\n';
            }
            std::cout << "dynamic=PASS\n";
        } catch (const std::exception& error) {
            std::cout << "dynamic=UNSUPPORTED:" << error.what() << '\n';
        }

        auto throughput_model = core.compile_model(
            model,
            device,
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        std::vector<ov::InferRequest> requests;
        for (size_t i = 0; i < 4; ++i) {
            requests.push_back(throughput_model.create_infer_request());
            requests.back().set_input_tensor(0, make_input({1024, 1024}, 0.2f));
            requests.back().set_input_tensor(1, make_input({1024, 1024}, 1.1f));
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
            try {
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
                std::cout << "cache_infer_ms=" << infer_once(cache_request, {1024, 1024}, &cache_checksum) << '\n';
                std::cout << "cache_checksum=" << std::setprecision(9) << cache_checksum << std::setprecision(3) << '\n';
                std::cout << "cache=" << (matches(static_checksum, cache_checksum) ? "PASS" : "FAIL") << '\n';
            } catch (const std::exception& error) {
                std::cout << "cache=UNSUPPORTED:" << error.what() << '\n';
            }
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
