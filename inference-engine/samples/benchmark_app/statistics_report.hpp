// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <utility>
#include <map>

#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/csv_dumper.hpp>

// @brief statistics reports types
static constexpr char noCntReport[] = "no_counters";
static constexpr char averageCntReport[] = "average_counters";
static constexpr char detailedCntReport[] = "detailed_counters";

/// @brief Responsible for collecting of statistics and dumping to .csv file
class StatisticsReport {
public:
    struct Config {
        std::string device;
        std::string api;
        size_t batch;
        size_t nireq;
        size_t niter;
        uint64_t duration;
        size_t cpu_nthreads;
        std::map<std::string, uint32_t> nstreams;
        std::string cpu_pin;
        std::string report_type;
        std::string report_folder;
    };

    explicit StatisticsReport(Config config) : _config(std::move(config)) {
        if (_config.nireq > 0) {
            _performanceCounters.reserve(_config.nireq);
        }
    }

    void addPerfCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &pmStat);

    void addLatencies(const std::vector<double> &latency);

    void dump(const double &fps, const size_t &numProcessedReq, const double &totalExecTime);

    double getMedianLatency();

private:
    std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> preparePmStatistics();

    template <typename T>
    T getMedianValue(const std::vector<T> &vec);

    // Contains PM data for each processed infer request
    std::vector<std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>> _performanceCounters;
    // Contains latency of each processed infer request
    std::vector<double> _latencies;

    // configuration of current benchmark execution
    const Config _config;

    // mapping from network layer to a vector of calculated RealTime values from each processed infer request.
    std::map<std::string, std::vector<long long>> _perLayerRealTime;
    // mapping from network layer to a vector of calculated CPU Time values from each processed infer request.
    std::map<std::string, std::vector<long long>> _perLayerCpuTime;
    std::vector<long long> _totalLayersTime;
};
