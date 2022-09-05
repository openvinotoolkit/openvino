// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <iostream>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest {
    static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

public:
    void Run(const std::initializer_list<std::string>& names,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        bench_names_ = names;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        LayerTestsUtils::LayerTestsCommon::configuration = {{"PERF_COUNT", "YES"}};
        LayerTestsUtils::LayerTestsCommon::Run();
    }

    void Run(const std::string& name,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        if (!name.empty()) {
            Run({name}, warmupTime, numAttempts);
        } else {
            Run({}, warmupTime, numAttempts);
        }
    }

    void Validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }

protected:
    void Infer() override {
        this->inferRequest = this->executableNetwork.CreateInferRequest();
        this->ConfigureInferRequest();

        std::map<std::string, long long> results_us{};
        for (const auto& name : bench_names_) {
            results_us[name] = {};
        }

        // Warmup
        auto warmCur = std::chrono::steady_clock::now();
        const auto warmEnd = warmCur + warmup_time_;
        while (warmCur < warmEnd) {
            this->inferRequest.Infer();
            warmCur = std::chrono::steady_clock::now();
        }

        // Benchmark
        for (int i = 0; i < num_attempts_; ++i) {
            this->inferRequest.Infer();
            const auto& perf_results = LayerTestsUtils::LayerTestsCommon::inferRequest.GetPerformanceCounts();
            for (auto& res : results_us) {
                const std::string name = res.first;
                long long& time = res.second;
                auto found_profile = std::find_if(perf_results.begin(), perf_results.end(),
                    [&name](const InferenceEngine::InferenceEngineProfileInfo& profile) {
                        return profile.layer_type == name;
                    });
                assert(found_profile != perf_results.end());
                time += found_profile->second.realTime_uSec;
            }
        }

        long long total_us = 0;
        for (auto& res : results_us) {
            const std::string name = res.first;
            long long& time = res.second;
            time /= num_attempts_;
            total_us += time;
            std::cout << std::fixed << std::setfill('0') << name << ": " << time << " us\n";
        }
        std::cout << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
    }

private:
    std::vector<std::string> bench_names_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};

}  // namespace LayerTestsDefinitions

namespace ov {
namespace test {

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest {
    static_assert(std::is_base_of<SubgraphBaseTest, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

 public:
    void run(const std::initializer_list<std::string>& names,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        bench_names_ = names;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        SubgraphBaseTest::configuration = {{"PERF_COUNT", "YES"}};
        SubgraphBaseTest::run();
    }

    void run(const std::string& name,
             const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
             const int numAttempts = 100) {
        if (!name.empty()) {
            run({name}, warmupTime, numAttempts);
        } else {
            run({}, warmupTime, numAttempts);
        }
    }

    void validate() override {
        // NOTE: Validation is ignored because we are interested in benchmarks results
    }

 protected:
    void infer() override {
        this->inferRequest = this->compiledModel.create_infer_request();
        for (const auto& input : this->inputs) {
            this->inferRequest.set_tensor(input.first, input.second);
        }

        std::map<std::string, long long> results_us{};
        for (const auto& name : bench_names_) {
            results_us[name] = {};
        }

        // Warmup
        auto warmCur = std::chrono::steady_clock::now();
        const auto warmEnd = warmCur + warmup_time_;
        while (warmCur < warmEnd) {
            this->inferRequest.infer();
            warmCur = std::chrono::steady_clock::now();
        }

        // Benchmark
        for (int i = 0; i < num_attempts_; ++i) {
            this->inferRequest.infer();
            const auto& profiling_info = SubgraphBaseTest::inferRequest.get_profiling_info();
            for (auto& res : results_us) {
                const std::string name = res.first;
                long long& time = res.second;
                auto found_profile = std::find_if(profiling_info.begin(), profiling_info.end(),
                    [&name](const ProfilingInfo& profile) {
                        return profile.node_type == name;
                    });
                assert(found_profile != profiling_info.end());
                time += found_profile->real_time.count();
            }
        }

        long long total_us = 0;
        for (auto& res : results_us) {
            const std::string name = res.first;
            long long& time = res.second;
            time /= num_attempts_;
            total_us += time;
            std::cout << std::fixed << std::setfill('0') << name << ": " << time << " us\n";
        }
        std::cout << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
    }

 private:
    std::vector<std::string> bench_names_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};

} // namespace test
} // namespace ov
