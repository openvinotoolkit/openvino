// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>

#include "statistics_report.hpp"

void StatisticsReport::addParameters(const Category &category, const Parameters& parameters) {
    if (_parameters.count(category) == 0)
        _parameters[category] = parameters;
    else
        _parameters[category].insert(_parameters[category].end(), parameters.begin(), parameters.end());
}

void StatisticsReport::dump() {
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_report.csv");

    auto dump_parameters = [ &dumper ] (const Parameters &parameters) {
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

void StatisticsReport::dumpPerformanceCountersRequest(CsvDumper& dumper,
                                                      const PerformaceCounters& perfCounts) {
    auto performanceMapSorted = perfCountersSorted(perfCounts);

    long long total = 0L;
    long long total_cpu = 0L;

    dumper << "layerName" << "execStatus" << "layerType" << "execType";
    dumper << "realTime (ms)" << "cpuTime (ms)";
    dumper.endLine();

    for (const auto &layer : performanceMapSorted) {
        dumper << layer.first;  // layer name

        switch (layer.second.status) {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                dumper << "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                dumper << "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                dumper << "OPTIMIZED_OUT";
                break;
        }
        dumper << layer.second.layer_type << layer.second.exec_type;
        dumper << std::to_string(layer.second.realTime_uSec / 1000.0) << std::to_string(layer.second.cpu_uSec/ 1000.0);
        total += layer.second.realTime_uSec;
        total_cpu += layer.second.cpu_uSec;
        dumper.endLine();
    }
    dumper << "Total" << "" << "" << "";
    dumper <<  total / 1000.0 << total_cpu / 1000.0;
    dumper.endLine();
    dumper.endLine();
}

void StatisticsReport::dumpPerformanceCounters(const std::vector<PerformaceCounters> &perfCounts) {
    if ((_config.report_type.empty()) || (_config.report_type == noCntReport)) {
        slog::info << "Statistics collecting for performance counters was not requested. No reports are dumped." << slog::endl;
        return;
    }
    if (perfCounts.empty()) {
        slog::info << "Peformance counters are empty. No reports are dumped." << slog::endl;
        return;
    }
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_" + _config.report_type + "_report.csv");
    if (_config.report_type == detailedCntReport) {
        for (auto& pc : perfCounts) {
            dumpPerformanceCountersRequest(dumper, pc);
        }
    } else if (_config.report_type == averageCntReport) {
        auto getAveragePerformanceCounters = [ &perfCounts ] () {
            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> performanceCountersAvg;
            // sort PM data of first processed request according to layers execution order
            auto performanceMapSorted = perfCountersSorted(perfCounts[0]);

            // iterate over each processed infer request and handle its PM data
            for (size_t i = 0; i < perfCounts.size(); i++) {
                // iterate over each layer from sorted vector and add required PM data to the per-layer maps
                for (const auto& pm : performanceMapSorted) {
                    if (performanceCountersAvg.count(pm.first) == 0) {
                        performanceCountersAvg[pm.first] = perfCounts.at(i).at(pm.first);
                    } else {
                        performanceCountersAvg[pm.first].realTime_uSec += perfCounts.at(i).at(pm.first).realTime_uSec;
                        performanceCountersAvg[pm.first].cpu_uSec += perfCounts.at(i).at(pm.first).cpu_uSec;
                    }
                }
            }
            for (auto& pm : performanceCountersAvg) {
                pm.second.realTime_uSec /= perfCounts.size();
                pm.second.cpu_uSec /= perfCounts.size();
            }
            return performanceCountersAvg;
        };
        dumpPerformanceCountersRequest(dumper, getAveragePerformanceCounters());
    } else {
        throw std::logic_error("PM data can only be collected for average or detailed report types");
    }
    slog::info << "Pefromance counters report is stored to " << dumper.getFilename() << slog::endl;
}
