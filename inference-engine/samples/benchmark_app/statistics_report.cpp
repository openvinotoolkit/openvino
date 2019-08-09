// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>

#include "statistics_report.hpp"

void StatisticsReport::addPerfCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &pmStat) {
    if (_config.report_type == averageCntReport || _config.report_type == detailedCntReport) {
        // collect per-iteration statistics only in case of enabled median/detailed statistic collecting
        _performanceCounters.push_back(pmStat);
    }
}

void StatisticsReport::addLatencies(const std::vector<double> &latencies) {
    _latencies.insert(_latencies.end(), latencies.begin(), latencies.end());
}

void StatisticsReport::dump(const double &fps, const size_t &iteration_number, const double &totalExecTime) {
    if (_config.report_type.empty()) {
        slog::info << "Statistics collecting was not requested. No reports are dumped." << slog::endl;
        return;
    }

    std::string separator =
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
        separator = "";

    CsvDumper dumper(true, _config.report_folder + separator + "benchmark_" + _config.report_type + "_report.csv");

    // resulting number of columns in csv file depends on the report_type. If it's noCntReport, then
    // no PM data is collected and there are only 3 columns in the file (in configuration section). If it's
    // averageCntReport then median PM values are collected per each layer and the number of columns is 6.
    // Example from GPU:
    //
    // layer name;exec status;layer type;exec type;real time;cpu time;
    // conv1;EXECUTED;Convolution;convolution_gpu_bfyx_gemm_like;615;3;
    // Here, all the data are taken from InferenceEngine::InferenceEngineProfileInfo.
    //
    // In case of detailedCntReport the number of columns is 4 + _config.nireq * 2, because first 4 parameters
    // are the same but realTime and cpuTime can be different on each iteration (example from 5 GPU requests):
    // conv1;EXECUTED;Convolution;convolution_gpu_bfyx_gemm_like;630,3;617,3;616,3;615,3;617,3;
    size_t numOfColumns = 0;
    if (_config.report_type == noCntReport) {
        numOfColumns = 3;
    } else if (_config.report_type == averageCntReport) {
        numOfColumns = 6;
    } else {
        // for detailedCntReport
        numOfColumns = 4 + _config.nireq * 2;
    }

    auto completeCsvRow = [](CsvDumper &dumper, size_t numOfColumns, size_t filled) {
        for (size_t i = 0; i < numOfColumns - filled; i++)
            dumper << "";
        dumper.endLine();
    };

    // dump execution configuration
    dumper << "Configuration setup";
    completeCsvRow(dumper, numOfColumns, 1);
    dumper << "config option" << "CLI parameter" << "value";
    completeCsvRow(dumper, numOfColumns, 3);

    dumper << "target device" << " -d" << _config.device;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "execution mode" << " -api" << _config.api;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "batch size" << " -b" << _config.batch;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "number of iterations" << " -niter" << _config.niter;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "number of parallel infer requests" << " -nireq" << _config.nireq;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "duration in ms" << " -t" << _config.duration;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "number of CPU threads" << " -nthreads" << _config.cpu_nthreads;
    completeCsvRow(dumper, numOfColumns, 3);
    for (auto& item : _config.nstreams)
        dumper << "number of " << item.first << " streams" << " -nstreams" << item.second;
    completeCsvRow(dumper, numOfColumns, 3);
    dumper << "CPU pinning enabled" << " -pin" << _config.cpu_pin;
    completeCsvRow(dumper, numOfColumns, 3);

    dumper.endLine();

    // write PM data from each iteration
    if (!_performanceCounters.empty()) {
        if (_config.report_type != averageCntReport && _config.report_type != detailedCntReport) {
            throw std::logic_error("PM data can only be collected for average or detailed report types");
        }

        // this vector is sorted according to network layers execution order.
        auto performanceMapSorted = preparePmStatistics();

        dumper << "Performance counters";
        completeCsvRow(dumper, numOfColumns, 1);
        dumper << "layer name" << "exec status" << "layer type" << "exec type";

        if (_config.report_type == averageCntReport) {
            dumper << "average real time" << "average cpu time";
            completeCsvRow(dumper, numOfColumns, 6);
        } else {
            // detailedCntReport case
            for (size_t i = 0; i< _performanceCounters.size(); i++) {
                dumper << "realTime_req" + std::to_string(i) << "cpuTime_req" + std::to_string(i);
            }
            completeCsvRow(dumper, numOfColumns, 4 + _performanceCounters.size() * 2);
        }

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

            if (_config.report_type == averageCntReport) {
                // write average realTime and cpuTime from each processed request for current layer
                dumper <<
                std::to_string(std::accumulate(_perLayerRealTime[layer.first].begin(),
                                               _perLayerRealTime[layer.first].end(), 0.0) / _perLayerRealTime[layer.first].size() / 1000.0) <<
                std::to_string(std::accumulate(_perLayerCpuTime[layer.first].begin(),
                                               _perLayerCpuTime[layer.first].end(), 0.0) / _perLayerCpuTime[layer.first].size()  / 1000.0);
            } else {
                // write all realTime and cpuTime from each processed request for current layer
                for (size_t i = 0; i < _config.nireq; i++) {
                    dumper << std::to_string(_perLayerRealTime[layer.first][i] / 1000.0) << std::to_string(_perLayerCpuTime[layer.first][i] / 1000.0);
                }
            }
            dumper.endLine();
        }
        dumper.endLine();
    }

    if (_config.report_type == detailedCntReport) {
        dumper << "Statistics";
        completeCsvRow(dumper, numOfColumns, 1);

        dumper << "metric";
        for (size_t i = 0; i < _totalLayersTime.size(); i++) {
            // detailedCntReport case
            dumper << "req" + std::to_string(i);
        }
        completeCsvRow(dumper, numOfColumns, 4 + _totalLayersTime.size());
        dumper << "latencies";
        for (const auto &lat : _totalLayersTime) {
            dumper << lat / 1000.0;
        }
        completeCsvRow(dumper, numOfColumns, _totalLayersTime.size());
        dumper.endLine();
    }

    dumper << "Execution results";
    completeCsvRow(dumper, numOfColumns, 1);
    dumper << "number of iterations" << iteration_number;
    completeCsvRow(dumper, numOfColumns, 2);
    dumper << "latency" << getMedianValue<double>(_latencies);
    completeCsvRow(dumper, numOfColumns, 2);
    dumper << "throughput" << fps;
    completeCsvRow(dumper, numOfColumns, 2);
    dumper << "total execution time" << totalExecTime;
    completeCsvRow(dumper, numOfColumns, 2);

    slog::info << "statistics report is stored to " << dumper.getFilename() << slog::endl;
}

double StatisticsReport::getMedianLatency() {
    return getMedianValue<double>(_latencies);
}

std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> StatisticsReport::preparePmStatistics() {
    if (_performanceCounters.empty()) {
        throw std::logic_error("preparePmStatistics() was called when no PM data was collected");
    }

    // sort PM data of first processed request according to layers execution order
    auto performanceMapSorted = perfCountersSorted(_performanceCounters[0]);

    // iterate over each processed infer request and handle its PM data
    for (auto &pm : _performanceCounters) {
        long long total = 0L;
        // iterate over each layer from sorted vector and add required PM data to the per-layer maps
        for (const auto & it : performanceMapSorted) {
            _perLayerRealTime[it.first].push_back(pm[it.first].realTime_uSec);
            _perLayerCpuTime[it.first].push_back(pm[it.first].cpu_uSec);
            total += pm[it.first].realTime_uSec;
        }
        _totalLayersTime.push_back(total);
    }
    return performanceMapSorted;
}

template <typename T>
T StatisticsReport::getMedianValue(const std::vector<T> &vec) {
    std::vector<T> sortedVec(vec);
    std::sort(sortedVec.begin(), sortedVec.end());
    return (sortedVec.size() % 2 != 0) ?
           sortedVec[sortedVec.size() / 2ULL] :
           (sortedVec[sortedVec.size() / 2ULL] + sortedVec[sortedVec.size() / 2ULL - 1ULL]) / static_cast<T>(2.0);
}
