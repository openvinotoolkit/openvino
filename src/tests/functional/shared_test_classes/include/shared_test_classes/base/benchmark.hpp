// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <iostream>

#include "pugixml.hpp"
#include "layer_test_utils.hpp"
#include "ov_subgraph.hpp"

namespace ov {
namespace test {

class BenchmarkLayerTestReporter {
public:
    static constexpr const char* const benchmarkReportFileName = "benchmark_layers.xml";
    static constexpr const char* const timeAttributeName = "time";

    explicit BenchmarkLayerTestReporter(bool is_readonly) : is_readonly_{is_readonly} {
        report_xml_.load_file(benchmarkReportFileName);
    }

    ~BenchmarkLayerTestReporter() {
        if (!is_readonly_) {
            report_xml_.save_file(benchmarkReportFileName);
        }
    }

    void report(const std::string& nodeTypeName, const std::string& testCaseName, const uint64_t time) {
        pugi::xml_node nodeTypeNode = report_xml_.child(nodeTypeName.c_str());
        if (!nodeTypeNode) {
            nodeTypeNode = report_xml_.append_child(nodeTypeName.c_str());
        }

        pugi::xml_node testCaseNode = nodeTypeNode.child(testCaseName.c_str());
        if (!testCaseNode) {
            testCaseNode = nodeTypeNode.append_child(testCaseName.c_str());
        }

        pugi::xml_attribute timeAttribute = testCaseNode.attribute(timeAttributeName);
        if (!timeAttribute) {
            timeAttribute = testCaseNode.append_attribute(timeAttributeName);
        }

        timeAttribute.set_value(static_cast<unsigned long long>(time));
    }

    uint64_t get_time(const std::string& nodeTypeName, const std::string& testCaseName) {
        pugi::xml_attribute timeAttribute =
            report_xml_.child(nodeTypeName.c_str()).child(testCaseName.c_str()).attribute(timeAttributeName);
        if (!timeAttribute) {
            throw std::range_error("no time stored for " + testCaseName);
        }

        return std::stoull(timeAttribute.value());
    }

private:
    bool is_readonly_{};
    pugi::xml_document report_xml_{};
};

}  // namespace test
}  // namespace ov

namespace LayerTestsDefinitions {

template <typename BaseLayerTest>
class BenchmarkLayerTest : public BaseLayerTest {
    static_assert(std::is_base_of<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>::value,
                  "BaseLayerTest should inherit from LayerTestsUtils::LayerTestsCommon");

public:
    static constexpr int kDefaultNumberOfAttempts = 100;
    static constexpr double kMaxAllowedBenchmarkDifference = 0.05;

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

    void Validate() override {
        for (const auto& res : curr_bench_results_) {
            const auto& node_type_name = res.first;
            const auto curr_time = static_cast<int64_t>(res.second);
            if (prev_bench_results_.count(node_type_name) > 0) {
                const auto prev_time = static_cast<int64_t>(prev_bench_results_[node_type_name]);
                const auto delta_time = static_cast<double>(curr_time - prev_time);
                if (delta_time/prev_time > kMaxAllowedBenchmarkDifference) {
                    std::cerr << "node_type_name: " << node_type_name <<
                                 ", for test case: " << BaseLayerTest::GetTestName() <<
                                 ", has exceeded the benchmark threshold: " << kMaxAllowedBenchmarkDifference <<
                                 ". Current: " << curr_time << " us, previous: " << prev_time << " us" << std::endl;
                }
            }
        }
    }

protected:
    void Infer() override {
        this->inferRequest = this->executableNetwork.CreateInferRequest();
        this->ConfigureInferRequest();

#ifdef ENABLE_BENCHMARK_FILE_REPORT
        reporter_ = std::unique_ptr<ov::test::BenchmarkLayerTestReporter>(
                new ::ov::test::BenchmarkLayerTestReporter{false});
#else
        reporter_ = std::unique_ptr<ov::test::BenchmarkLayerTestReporter>(
                new ::ov::test::BenchmarkLayerTestReporter{true});
#endif
        for (const auto& node_type_name : bench_node_type_names_) {
            try {
                const auto time = reporter_->get_time(node_type_name, BaseLayerTest::GetTestName());
                prev_bench_results_[node_type_name] = time;
            } catch (...) {
            }
        }

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
#ifdef ENABLE_BENCHMARK_FILE_REPORT
            curr_bench_results_[node_type_name] = time;
            reporter_->report(node_type_name, BaseLayerTest::GetTestName(), time);
#endif
        }
        report << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
        std::cout << report.str();
    }

private:
    std::unique_ptr<ov::test::BenchmarkLayerTestReporter> reporter_;
    std::unordered_map<std::string, uint64_t> prev_bench_results_;
    std::unordered_map<std::string, uint64_t> curr_bench_results_;
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
    static constexpr double kMaxAllowedBenchmarkDifference = 0.05;

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

    void validate() override {
        for (const auto& res : curr_bench_results_) {
            const auto& node_type_name = res.first;
            const auto curr_time = static_cast<int64_t>(res.second);
            if (prev_bench_results_.count(node_type_name) > 0) {
                const auto prev_time = static_cast<int64_t>(prev_bench_results_[node_type_name]);
                const auto delta_time = static_cast<double>(curr_time - prev_time);
                if (delta_time/prev_time > kMaxAllowedBenchmarkDifference) {
                    std::cerr << "node_type_name: " << node_type_name <<
                              ", for test case: " << BaseLayerTest::GetTestName() <<
                              ", has exceeded the benchmark threshold: " << kMaxAllowedBenchmarkDifference <<
                              ". Current: " << curr_time << " us, previous: " << prev_time << " us" << std::endl;
                }
            }
        }
    }

 protected:
    void infer() override {
        this->inferRequest = this->compiledModel.create_infer_request();
        for (const auto& input : this->inputs) {
            this->inferRequest.set_tensor(input.first, input.second);
        }

#ifdef ENABLE_BENCHMARK_FILE_REPORT
        reporter_ = std::unique_ptr<ov::test::BenchmarkLayerTestReporter>(
                new ::ov::test::BenchmarkLayerTestReporter{false});
#else
        reporter_ = std::unique_ptr<ov::test::BenchmarkLayerTestReporter>(
                new ::ov::test::BenchmarkLayerTestReporter{true});
#endif
        for (const auto& node_type_name : bench_node_type_names_) {
            try {
                const auto time = reporter_->get_time(node_type_name, BaseLayerTest::GetTestName());
                prev_bench_results_[node_type_name] = time;
            } catch (...) {
            }
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
        for (int i = 0; i < num_attempts_; ++i) {
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
#ifdef ENABLE_BENCHMARK_FILE_REPORT
            curr_bench_results_[node_type_name] = time;
            reporter_->report(node_type_name, BaseLayerTest::GetTestName(), time);
#endif
        }
        report << std::fixed << std::setfill('0') << "Total time: " << total_us << " us\n";
        std::cout << report.str();
    }

 private:
    std::unique_ptr<ov::test::BenchmarkLayerTestReporter> reporter_;
    std::unordered_map<std::string, uint64_t> prev_bench_results_;
    std::unordered_map<std::string, uint64_t> curr_bench_results_;
    std::vector<std::string> bench_node_type_names_;
    std::chrono::milliseconds warmup_time_;
    int num_attempts_;
};

}  // namespace test
}  // namespace ov
