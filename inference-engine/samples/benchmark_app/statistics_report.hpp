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
    typedef std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> PerformaceCounters;
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
    #   if defined UNICODE
        L"\\";
    #   else
        "\\";
    #   endif
#else
        "/";
#endif
        if (_config.report_folder.empty())
            _separator = "";
    }

    void addParameters(const Category &category, const Parameters& parameters);

    void dump();

    void dumpPerformanceCounters(const std::vector<PerformaceCounters> &perfCounts);

private:
    void dumpPerformanceCountersRequest(CsvDumper& dumper,
                                        const PerformaceCounters& perfCounts);

    // configuration of current benchmark execution
    const Config _config;

    // parameters
    std::map<Category, Parameters> _parameters;

    // csv separator
    std::string _separator;
};
