// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "samples/common.hpp"
#include "samples/csv_dumper.hpp"
#include "samples/slog.hpp"

#include "utils.hpp"
// clang-format on

// @brief statistics reports types
static constexpr char noCntReport[] = "no_counters";
static constexpr char averageCntReport[] = "average_counters";
static constexpr char detailedCntReport[] = "detailed_counters";

/// @brief Responsible for calculating different latency metrics
class LatencyMetrics {
public:
    LatencyMetrics() = delete;

    LatencyMetrics(const std::vector<double>& latencies) : latencies(latencies) {
        if (latencies.empty()) {
            throw std::logic_error("Latency metrics class expects non-empty vector of latencies at consturction.");
        }
        std::sort(this->latencies.begin(), this->latencies.end());
    }

    LatencyMetrics(std::vector<double>&& latencies) : latencies(latencies) {
        if (latencies.empty()) {
            throw std::logic_error("Latency metrics class expects non-empty vector of latencies at consturction.");
        }
        std::sort(this->latencies.begin(), this->latencies.end());
    }

    double min() {
        return latencies[0];
    }

    double average() {
        return std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    }

    double percentile(std::size_t p) {
        return latencies[size_t(latencies.size() / 100.0 * p)];
    }

    double max() {
        return latencies.back();
    }

    void log_total(size_t p) {
        std::string percentileStr = (p == 50) ? "\tMedian:  " : "\t" + std::to_string(p) + " percentile:    ";
        slog::info << percentileStr << double_to_string(percentile(p)) << " ms" << slog::endl;
        slog::info << "\tAvg:    " << double_to_string(average()) << " ms" << slog::endl;
        slog::info << "\tMin:    " << double_to_string(min()) << " ms" << slog::endl;
        slog::info << "\tMax:    " << double_to_string(max()) << " ms" << slog::endl;
    }

private:
    std::vector<double> latencies;
};

/// @brief Responsible for collecting of statistics and dumping to .csv file
class StatisticsReport {
public:
    typedef std::vector<ov::runtime::ProfilingInfo> PerformaceCounters;
    typedef std::vector<std::pair<std::string, std::string>> Parameters;

    struct Config {
        std::string report_type;
        std::string report_folder;
    };

    enum class Category {
        COMMAND_LINE_PARAMETERS,
        RUNTIME_CONFIG,
        EXECUTION_RESULTS,
    };

    explicit StatisticsReport(Config config) : _config(std::move(config)) {
        _separator =
#if defined _WIN32 || defined __CYGWIN__
#    if defined UNICODE
            L"\\";
#    else
            "\\";
#    endif
#else
            "/";
#endif
        if (_config.report_folder.empty())
            _separator = "";
    }

    void add_parameters(const Category& category, const Parameters& parameters);

    void dump();

    void dump_performance_counters(const std::vector<PerformaceCounters>& perfCounts);

private:
    void dump_performance_counters_request(CsvDumper& dumper, const PerformaceCounters& perfCounts);

    // configuration of current benchmark execution
    const Config _config;

    // parameters
    std::map<Category, Parameters> _parameters;

    // csv separator
    std::string _separator;
};
