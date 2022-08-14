// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <iostream>

#include "layer_test_utils.hpp"
#include "ov_subgraph.hpp"

namespace LayerTestsDefinitions {

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest {
    static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

public:
    static constexpr int kDefaultNumberOfAttempts = 100;

    void RunBenchmark(const std::initializer_list<std::string>& nodeTypeNames,
                      const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
                      const int numAttempts = kDefaultNumberOfAttempts) {
        bench_node_type_names_ = nodeTypeNames;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        this->configuration.insert({"PERF_COUNT", "YES"});
        this->Run();
    }

    void RunBenchmark(const std::string& nodeTypeName,
                      const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
                      const int numAttempts = kDefaultNumberOfAttempts) {
        if (!nodeTypeName.empty()) {
            RunBenchmark({nodeTypeName}, warmupTime, numAttempts);
        } else {
            RunBenchmark({}, warmupTime, numAttempts);
        }
    }

    // NOTE: Validation is ignored because we are interested in benchmarks results.
    //       In future the validate method could check if new benchmark results are not worse than previous one
    //       (regression test), and in case of any performance issue report it in PR.
    void Validate() override {
    }

protected:
    void Infer() override {
        this->inferRequest = this->executableNetwork.CreateInferRequest();
        this->ConfigureInferRequest();

        std::map<std::string, uint64_t> results_us{};
        for (const auto& node_type_name : bench_node_type_names_) {
            results_us[node_type_name] = {};
        }

        // Warmup
        auto warm_current = std::chrono::steady_clock::now();
        const auto warm_end = warm_current + warmup_time_;
        while (warm_current < warm_end) {
            this->inferRequest.Infer();
            warm_current = std::chrono::steady_clock::now();
        }

        // Benchmark
        for (size_t i = 0; i < num_attempts_; ++i) {
            this->inferRequest.Infer();
            const auto& perf_results = this->inferRequest.GetPerformanceCounts();
            for (auto& res : results_us) {
                const std::string node_type_name = res.first;
                uint64_t& time = res.second;
                auto found_profile = std::find_if(perf_results.begin(), perf_results.end(),
                    [&node_type_name](const InferenceEngine::InferenceEngineProfileInfo& profile) {
                        return profile.layer_type == node_type_name;
                    });
                if (found_profile == perf_results.end()) {
                    IE_THROW() << "Cannot find operator by node type: " << node_type_name;
                }
                time += found_profile->second.realTime_uSec;
            }
        }

        std::stringstream report{};
        uint64_t total_us = 0;
        for (const auto& res : results_us) {
            const std::string node_type_name = res.first;
            uint64_t time = res.second;
            time /= num_attempts_;
            total_us += time;
            report << std::fixed << std::setfill('0') << node_type_name << ": " << time << " us\n";
        }
        report << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
        std::cout << report.str();
    }

private:
    std::vector<std::string> bench_node_type_names_;
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
    static constexpr int kDefaultNumberOfAttempts = 100;

    void run_benchmark(const std::initializer_list<std::string>& nodeTypeNames,
                       const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
                       const int numAttempts = kDefaultNumberOfAttempts) {
        bench_node_type_names_ = nodeTypeNames;
        warmup_time_ = warmupTime;
        num_attempts_ = numAttempts;
        this->configuration.insert({"PERF_COUNT", "YES"});
        this->run();
    }

    void run_benchmark(const std::string& nodeTypeName,
                       const std::chrono::milliseconds warmupTime = std::chrono::milliseconds(2000),
                       const int numAttempts = kDefaultNumberOfAttempts) {
        if (!nodeTypeName.empty()) {
            run_benchmark({nodeTypeName}, warmupTime, numAttempts);
        } else {
            run_benchmark({}, warmupTime, numAttempts);
        }
    }

    // NOTE: Validation is ignored because we are interested in benchmarks results.
    //       In future the validate method could check if new benchmark results are not worse than previous one
    //       (regression test), and in case of any performance issue report it in PR.
    void validate() override {
    }

 protected:
    void infer() override {
        this->inferRequest = this->compiledModel.create_infer_request();
        for (const auto& input : this->inputs) {
            this->inferRequest.set_tensor(input.first, input.second);
        }

        std::map<std::string, uint64_t> results_us{};
        for (const auto& node_type_name : bench_node_type_names_) {
            results_us[node_type_name] = {};
        }

        // Warmup
        auto warm_current = std::chrono::steady_clock::now();
        const auto warm_end = warm_current + warmup_time_;
        while (warm_current < warm_end) {
            this->inferRequest.infer();
            warm_current = std::chrono::steady_clock::now();
        }

        // Benchmark
        for (size_t i = 0; i < num_attempts_; ++i) {
            this->inferRequest.infer();
            const auto& profiling_info = this->inferRequest.get_profiling_info();
            for (auto& res : results_us) {
                const std::string node_type_name = res.first;
                uint64_t& time = res.second;
                auto found_profile = std::find_if(profiling_info.begin(), profiling_info.end(),
                    [&node_type_name](const ProfilingInfo& profile) {
                        return profile.node_type == node_type_name;
                    });
                if (found_profile == profiling_info.end()) {
                    IE_THROW() << "Cannot find operator by node type: " << node_type_name;
                }
                time += found_profile->real_time.count();
            }
        }

        std::stringstream report{};
        uint64_t total_us = 0;
        for (const auto& res : results_us) {
            const std::string node_type_name = res.first;
            uint64_t time = res.second;
            time /= num_attempts_;
            total_us += time;
            report << std::fixed << std::setfill('0') << node_type_name << ": " << time << " us\n";
        }
        report << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
        std::cout << report.str();
    }

 private:
    std::vector<std::string> bench_node_type_names_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};

}  // namespace test
}  // namespace ov
