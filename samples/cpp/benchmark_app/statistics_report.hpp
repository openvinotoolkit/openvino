// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <nlohmann/json.hpp>
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
    LatencyMetrics() {}

    LatencyMetrics(const std::vector<double>& latencies,
                   const std::string& data_shape = "",
                   size_t percentile_boundary = 50)
        : data_shape(data_shape),
          percentile_boundary(percentile_boundary) {
        fill_data(latencies, percentile_boundary);
    }

    void write_to_stream(std::ostream& stream) const;
    void write_to_slog() const;
    const nlohmann::json to_json() const;

public:
    double median_or_percentile = 0;
    double avg = 0;
    double min = 0;
    double max = 0;
    std::string data_shape;

private:
    void fill_data(std::vector<double> latencies, size_t percentile_boundary);
    size_t percentile_boundary = 50;
};

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
};
