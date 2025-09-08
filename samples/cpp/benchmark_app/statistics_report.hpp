// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef JSON_HEADER
#    include <json.hpp>
#else
#    include <nlohmann/json.hpp>
#endif

// clang-format off
#include "samples/common.hpp"
#include "samples/csv_dumper.hpp"
#include "samples/slog.hpp"
#include "samples/latency_metrics.hpp"

#include "utils.hpp"
// clang-format on

// @brief statistics reports types
static constexpr char noCntReport[] = "no_counters";
static constexpr char averageCntReport[] = "average_counters";
static constexpr char detailedCntReport[] = "detailed_counters";
static constexpr char sortDetailedCntReport[] = "sort_detailed_counters";

class StatisticsVariant {
public:
    enum Type { INT, DOUBLE, STRING, ULONGLONG, METRICS };

    StatisticsVariant(std::string csv_name, std::string json_name, int v)
        : csv_name(csv_name),
          json_name(json_name),
          i_val(v),
          type(INT) {}
    StatisticsVariant(std::string csv_name, std::string json_name, double v)
        : csv_name(csv_name),
          json_name(json_name),
          d_val(v),
          type(DOUBLE) {}
    StatisticsVariant(std::string csv_name, std::string json_name, const std::string& v)
        : csv_name(csv_name),
          json_name(json_name),
          s_val(v),
          type(STRING) {}
    StatisticsVariant(std::string csv_name, std::string json_name, unsigned long long v)
        : csv_name(csv_name),
          json_name(json_name),
          ull_val(v),
          type(ULONGLONG) {}
    StatisticsVariant(std::string csv_name, std::string json_name, uint32_t v)
        : csv_name(csv_name),
          json_name(json_name),
          ull_val(v),
          type(ULONGLONG) {}
    StatisticsVariant(std::string csv_name, std::string json_name, unsigned long v)
        : csv_name(csv_name),
          json_name(json_name),
          ull_val(v),
          type(ULONGLONG) {}
    StatisticsVariant(std::string csv_name, std::string json_name, const LatencyMetrics& v)
        : csv_name(csv_name),
          json_name(json_name),
          metrics_val(v),
          type(METRICS) {}

    ~StatisticsVariant() {}

    std::string csv_name;
    std::string json_name;
    int i_val = 0;
    double d_val = 0;
    unsigned long long ull_val = 0;
    std::string s_val;
    LatencyMetrics metrics_val;
    Type type;

    std::string to_string() const;
    void write_to_json(nlohmann::json& js) const;
};

/// @brief Responsible for collecting of statistics and dumping to .csv file
class StatisticsReport {
public:
    typedef std::vector<ov::ProfilingInfo> PerformanceCounters;
    typedef std::vector<StatisticsVariant> Parameters;

    struct Config {
        std::string report_type;
        std::string report_folder;
    };

    enum class Category { COMMAND_LINE_PARAMETERS, RUNTIME_CONFIG, EXECUTION_RESULTS, EXECUTION_RESULTS_GROUPPED };

    virtual ~StatisticsReport() = default;

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

    virtual void dump();

    virtual void dump_performance_counters(const std::vector<PerformanceCounters>& perfCounts);

private:
    void dump_performance_counters_request(CsvDumper& dumper, const PerformanceCounters& perfCounts);
    void dump_sort_performance_counters_request(CsvDumper& dumper, const PerformanceCounters& perfCounts);
    static bool sort_profiling_descend(const ov::ProfilingInfo& profiling1, const ov::ProfilingInfo& profiling2) {
        return profiling1.real_time > profiling2.real_time;
    }

protected:
    // configuration of current benchmark execution
    const Config _config;

    // parameters
    std::map<Category, Parameters> _parameters;

    // csv separator
    std::string _separator;

    StatisticsReport::PerformanceCounters get_average_performance_counters(
        const std::vector<PerformanceCounters>& perfCounts);
};

class StatisticsReportJSON : public StatisticsReport {
public:
    explicit StatisticsReportJSON(Config config) : StatisticsReport(std::move(config)) {}

    void dump() override;
    void dump_performance_counters(const std::vector<PerformanceCounters>& perfCounts) override;

private:
    void dump_parameters(nlohmann::json& js, const StatisticsReport::Parameters& parameters);
    const nlohmann::json perf_counters_to_json(const StatisticsReport::PerformanceCounters& perfCounts);
    const nlohmann::json sort_perf_counters_to_json(const StatisticsReport::PerformanceCounters& perfCounts);
    static bool sort_profiling_descend(const ov::ProfilingInfo& profiling1, const ov::ProfilingInfo& profiling2) {
        return profiling1.real_time > profiling2.real_time;
    }
};
