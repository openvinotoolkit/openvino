// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "statistics_report.hpp"
// clang-format on

void StatisticsReport::add_parameters(const Category& category, const Parameters& parameters) {
    if (_parameters.count(category) == 0)
        _parameters[category] = parameters;
    else
        _parameters[category].insert(_parameters[category].end(), parameters.begin(), parameters.end());
}

void StatisticsReport::dump() {
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_report.csv");

    auto dump_parameters = [&dumper](const Parameters& parameters) {
        for (auto& parameter : parameters) {
            dumper << parameter.first << parameter.second;
            dumper.endLine();
        }
    };
    if (_parameters.count(Category::COMMAND_LINE_PARAMETERS)) {
        dumper << "Command line parameters";
        dumper.endLine();

        dump_parameters(_parameters.at(Category::COMMAND_LINE_PARAMETERS));
        dumper.endLine();
    }

    if (_parameters.count(Category::RUNTIME_CONFIG)) {
        dumper << "Configuration setup";
        dumper.endLine();

        dump_parameters(_parameters.at(Category::RUNTIME_CONFIG));
        dumper.endLine();
    }

    if (_parameters.count(Category::EXECUTION_RESULTS)) {
        dumper << "Execution results";
        dumper.endLine();

        dump_parameters(_parameters.at(Category::EXECUTION_RESULTS));
        dumper.endLine();
    }

    slog::info << "Statistics report is stored to " << dumper.getFilename() << slog::endl;
}

void StatisticsReport::dump_performance_counters_request(CsvDumper& dumper, const PerformaceCounters& perfCounts) {
    std::chrono::microseconds total = std::chrono::microseconds::zero();
    std::chrono::microseconds total_cpu = std::chrono::microseconds::zero();

    dumper << "layerName"
           << "execStatus"
           << "layerType"
           << "execType";
    dumper << "realTime (ms)"
           << "cpuTime (ms)";
    dumper.endLine();

    for (const auto& layer : perfCounts) {
        dumper << layer.node_name;  // layer name

        switch (layer.status) {
        case ov::runtime::ProfilingInfo::Status::EXECUTED:
            dumper << "EXECUTED";
            break;
        case ov::runtime::ProfilingInfo::Status::NOT_RUN:
            dumper << "NOT_RUN";
            break;
        case ov::runtime::ProfilingInfo::Status::OPTIMIZED_OUT:
            dumper << "OPTIMIZED_OUT";
            break;
        }
        dumper << layer.node_type << layer.exec_type;
        dumper << std::to_string(layer.real_time.count() / 1000.0) << std::to_string(layer.cpu_time.count() / 1000.0);
        total += layer.real_time;
        total_cpu += layer.cpu_time;
        dumper.endLine();
    }
    dumper << "Total"
           << ""
           << ""
           << "";
    dumper << total.count() / 1000.0 << total_cpu.count() / 1000.0;
    dumper.endLine();
    dumper.endLine();
}

void StatisticsReport::dump_performance_counters(const std::vector<PerformaceCounters>& perfCounts) {
    if ((_config.report_type.empty()) || (_config.report_type == noCntReport)) {
        slog::info << "Statistics collecting for performance counters was not "
                      "requested. No reports are dumped."
                   << slog::endl;
        return;
    }
    if (perfCounts.empty()) {
        slog::info << "Performance counters are empty. No reports are dumped." << slog::endl;
        return;
    }
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_" + _config.report_type + "_report.csv");
    if (_config.report_type == detailedCntReport) {
        for (auto& pc : perfCounts) {
            dump_performance_counters_request(dumper, pc);
        }
    } else if (_config.report_type == averageCntReport) {
        auto getAveragePerformanceCounters = [&perfCounts]() {
            std::vector<ov::runtime::ProfilingInfo> performanceCountersAvg;
            // iterate over each processed infer request and handle its PM data
            for (size_t i = 0; i < perfCounts.size(); i++) {
                // iterate over each layer from sorted vector and add required PM data
                // to the per-layer maps
                for (const auto& pm : perfCounts[i]) {
                    int idx = 0;
                    for (; idx < performanceCountersAvg.size(); idx++) {
                        if (performanceCountersAvg[idx].node_name == pm.node_name) {
                            performanceCountersAvg[idx].real_time += pm.real_time;
                            performanceCountersAvg[idx].cpu_time += pm.cpu_time;
                            break;
                        }
                    }
                    if (idx == performanceCountersAvg.size()) {
                        performanceCountersAvg.push_back(pm);
                    }
                }
            }
            for (auto& pm : performanceCountersAvg) {
                pm.real_time /= perfCounts.size();
                pm.cpu_time /= perfCounts.size();
            }
            return performanceCountersAvg;
        };
        dump_performance_counters_request(dumper, getAveragePerformanceCounters());
    } else {
        throw std::logic_error("PM data can only be collected for average or detailed report types");
    }
    slog::info << "Performance counters report is stored to " << dumper.getFilename() << slog::endl;
}
