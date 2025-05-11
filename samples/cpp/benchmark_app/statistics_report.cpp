// Copyright (C) 2018-2025 Intel Corporation
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

static const char* status_names[] = {"NOT_RUN", "OPTIMIZED_OUT", "EXECUTED"};

void StatisticsReport::add_parameters(const Category& category, const Parameters& parameters) {
    if (_parameters.count(category) == 0)
        _parameters[category] = parameters;
    else
        _parameters[category].insert(_parameters[category].end(), parameters.begin(), parameters.end());
}

void StatisticsReport::dump() {
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_report.csv", 3);

    auto dump_parameters = [&dumper](const Parameters& parameters) {
        for (auto& parameter : parameters) {
            if (parameter.type != StatisticsVariant::METRICS) {
                dumper << parameter.csv_name;
            }
            dumper << parameter.to_string();
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

    if (_parameters.count(Category::EXECUTION_RESULTS_GROUPPED)) {
        dumper << "Group Latencies";
        dumper.endLine();
        dumper << "Data shape;Median;Average;Min;Max";
        dumper.endLine();

        dump_parameters(_parameters.at(Category::EXECUTION_RESULTS_GROUPPED));
        dumper.endLine();
    }

    slog::info << "Statistics report is stored to " << dumper.getFilename() << slog::endl;
}

void StatisticsReport::dump_performance_counters_request(CsvDumper& dumper, const PerformanceCounters& perfCounts) {
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
        dumper << ((size_t)layer.status < (sizeof(status_names) / sizeof(status_names[0]))
                       ? status_names[(int)layer.status]
                       : "INVALID_STATUS");
        dumper << layer.node_type << layer.exec_type;
        dumper << layer.real_time.count() / 1000.0 << layer.cpu_time.count() / 1000.0;
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

void StatisticsReport::dump_sort_performance_counters_request(CsvDumper& dumper,
                                                              const PerformanceCounters& perfCounts) {
    std::chrono::microseconds total = std::chrono::microseconds::zero();
    std::chrono::microseconds total_cpu = std::chrono::microseconds::zero();

    dumper << "layerName"
           << "execStatus"
           << "layerType"
           << "execType";
    dumper << "realTime (ms)"
           << "cpuTime (ms)"
           << " %";
    dumper.endLine();

    for (const auto& layer : perfCounts) {
        if (std::string(status_names[(int)layer.status]).compare("EXECUTED") == 0) {
            total += layer.real_time;
            total_cpu += layer.cpu_time;
        }
    }

    // sort perfcounter
    std::vector<ov::ProfilingInfo> profiling{std::begin(perfCounts), std::end(perfCounts)};
    std::sort(profiling.begin(), profiling.end(), sort_profiling_descend);
    for (const auto& layer : profiling) {
        if (std::string(status_names[(int)layer.status]).compare("EXECUTED") == 0) {
            dumper << layer.node_name;  // layer name
            dumper << ((size_t)layer.status < (sizeof(status_names) / sizeof(status_names[0]))
                           ? status_names[(int)layer.status]
                           : "INVALID_STATUS");
            dumper << layer.node_type << layer.exec_type;
            dumper << layer.real_time.count() / 1000.0 << layer.cpu_time.count() / 1000.0;
            dumper << (layer.real_time * 1.0 / total) * 100;
            dumper.endLine();
        }
    }

    dumper << "Total"
           << ""
           << ""
           << "";
    dumper << total.count() / 1000.0 << total_cpu.count() / 1000.0 << 100.0;
    dumper.endLine();
    dumper.endLine();
}

StatisticsReport::PerformanceCounters StatisticsReport::get_average_performance_counters(
    const std::vector<PerformanceCounters>& perfCounts) {
    StatisticsReport::PerformanceCounters performanceCountersAvg;
    // iterate over each processed infer request and handle its PM data
    for (size_t i = 0; i < perfCounts.size(); i++) {
        // iterate over each layer from sorted vector and add required PM data
        // to the per-layer maps
        for (const auto& pm : perfCounts[i]) {
            size_t idx = 0;
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

void StatisticsReport::dump_performance_counters(const std::vector<PerformanceCounters>& perfCounts) {
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
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_" + _config.report_type + "_report.csv", 3);
    if (_config.report_type == detailedCntReport) {
        for (auto& pc : perfCounts) {
            dump_performance_counters_request(dumper, pc);
        }
    } else if (_config.report_type == averageCntReport) {
        dump_performance_counters_request(dumper, get_average_performance_counters(perfCounts));
    } else if (_config.report_type == sortDetailedCntReport) {
        for (auto& pc : perfCounts) {
            dump_sort_performance_counters_request(dumper, pc);
        }
    } else {
        throw std::logic_error("PM data can only be collected for average or detailed report types");
    }
    slog::info << "Performance counters report is stored to " << dumper.getFilename() << slog::endl;
}

void StatisticsReportJSON::dump_parameters(nlohmann::json& js, const StatisticsReport::Parameters& parameters) {
    for (auto& parameter : parameters) {
        parameter.write_to_json(js);
    }
};

void StatisticsReportJSON::dump() {
    nlohmann::json js;
    std::string name = _config.report_folder + _separator + "benchmark_report.json";

    if (_parameters.count(Category::COMMAND_LINE_PARAMETERS)) {
        dump_parameters(js["cmd_options"], _parameters.at(Category::COMMAND_LINE_PARAMETERS));
    }
    if (_parameters.count(Category::RUNTIME_CONFIG)) {
        dump_parameters(js["configuration_setup"], _parameters.at(Category::RUNTIME_CONFIG));
    }
    if (_parameters.count(Category::EXECUTION_RESULTS)) {
        dump_parameters(js["execution_results"], _parameters.at(Category::EXECUTION_RESULTS));
    }
    if (_parameters.count(Category::EXECUTION_RESULTS_GROUPPED)) {
        dump_parameters(js["execution_results"], _parameters.at(Category::EXECUTION_RESULTS_GROUPPED));
    }

    std::ofstream out_stream(name);
    out_stream << std::setw(4) << js << std::endl;
    slog::info << "Statistics report is stored to " << name << slog::endl;
}

void StatisticsReportJSON::dump_performance_counters(const std::vector<PerformanceCounters>& perfCounts) {
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

    nlohmann::json js;
    std::string name = _config.report_folder + _separator + "benchmark_" + _config.report_type + "_report.json";

    if (_config.report_type == detailedCntReport) {
        js["report_type"] = "detailed";
        js["detailed_performance"] = nlohmann::json::array();
        for (auto& pc : perfCounts) {
            js["detailed_performance"].push_back(perf_counters_to_json(pc));
        }
    } else if (_config.report_type == averageCntReport) {
        js["report_type"] = "average";
        js["avg_performance"] = perf_counters_to_json(get_average_performance_counters(perfCounts));
    } else if (_config.report_type == sortDetailedCntReport) {
        for (auto& pc : perfCounts) {
            js["detailed_performance"].push_back(sort_perf_counters_to_json(pc));
        }
    } else {
        throw std::logic_error("PM data can only be collected for average or detailed report types");
    }

    std::ofstream out_stream(name);
    out_stream << std::setw(4) << js << std::endl;
    slog::info << "Performance counters report is stored to " << name << slog::endl;
}

const nlohmann::json StatisticsReportJSON::perf_counters_to_json(
    const StatisticsReport::PerformanceCounters& perfCounts) {
    std::chrono::microseconds total = std::chrono::microseconds::zero();
    std::chrono::microseconds total_cpu = std::chrono::microseconds::zero();

    nlohmann::json js;
    js["nodes"] = nlohmann::json::array();
    for (const auto& layer : perfCounts) {
        nlohmann::json item;

        item["name"] = layer.node_name;  // layer name
        item["status"] =
            ((size_t)layer.status < (sizeof(status_names) / sizeof(status_names[0])) ? status_names[(int)layer.status]
                                                                                     : "INVALID_STATUS");
        item["node_type"] = layer.node_type;
        item["exec_type"] = layer.exec_type;
        item["real_time"] = layer.real_time.count() / 1000.0;
        item["cpu_time"] = layer.cpu_time.count() / 1000.0;
        total += layer.real_time;
        total_cpu += layer.cpu_time;
        js["nodes"].push_back(item);
    }
    js["total_real_time"] = total.count() / 1000.0;
    js["total_cpu_time"] = total_cpu.count() / 1000.0;
    return js;
}

const nlohmann::json StatisticsReportJSON::sort_perf_counters_to_json(
    const StatisticsReport::PerformanceCounters& perfCounts) {
    std::chrono::microseconds total = std::chrono::microseconds::zero();
    std::chrono::microseconds total_cpu = std::chrono::microseconds::zero();

    nlohmann::json js;
    js["nodes"] = nlohmann::json::array();

    for (const auto& layer : perfCounts) {
        total += layer.real_time;
        total_cpu += layer.cpu_time;
    }

    // sort perfcounter
    std::vector<ov::ProfilingInfo> sortPerfCounts{std::begin(perfCounts), std::end(perfCounts)};
    std::sort(sortPerfCounts.begin(), sortPerfCounts.end(), sort_profiling_descend);

    for (const auto& layer : sortPerfCounts) {
        nlohmann::json item;
        item["name"] = layer.node_name;  // layer name
        item["status"] =
            ((size_t)layer.status < (sizeof(status_names) / sizeof(status_names[0])) ? status_names[(int)layer.status]
                                                                                     : "INVALID_STATUS");
        item["node_type"] = layer.node_type;
        item["exec_type"] = layer.exec_type;
        item["real_time"] = layer.real_time.count() / 1000.0;
        item["cpu_time"] = layer.cpu_time.count() / 1000.0;
        item["%"] = std::round(layer.real_time * 10000.0 / total) / 100;
        js["nodes"].push_back(item);
    }

    js["total_real_time"] = total.count() / 1000.0;
    js["total_cpu_time"] = total_cpu.count() / 1000.0;
    return js;
}

static nlohmann::json to_json(const LatencyMetrics& latenct_metrics) {
    nlohmann::json stat;
    stat["data_shape"] = latenct_metrics.data_shape;
    stat["latency_median"] = latenct_metrics.median_or_percentile;
    stat["latency_average"] = latenct_metrics.avg;
    stat["latency_min"] = latenct_metrics.min;
    stat["latency_max"] = latenct_metrics.max;
    return stat;
}

std::string StatisticsVariant::to_string() const {
    switch (type) {
    case INT:
        return std::to_string(i_val);
    case DOUBLE:
        return std::to_string(d_val);
    case STRING:
        return s_val;
    case ULONGLONG:
        return std::to_string(ull_val);
    case METRICS:
        std::ostringstream str;
        metrics_val.write_to_stream(str);
        return str.str();
    }
    throw std::invalid_argument("StatisticsVariant::to_string : invalid type is provided");
}

void StatisticsVariant::write_to_json(nlohmann::json& js) const {
    switch (type) {
    case INT:
        js[json_name] = i_val;
        break;
    case DOUBLE:
        js[json_name] = d_val;
        break;
    case STRING:
        js[json_name] = s_val;
        break;
    case ULONGLONG:
        js[json_name] = ull_val;
        break;
    case METRICS: {
        auto& arr = js[json_name];
        if (arr.empty()) {
            arr = nlohmann::json::array();
        }
        arr.push_back(to_json(metrics_val));
    } break;
    default:
        throw std::invalid_argument("StatisticsVariant:: json conversion : invalid type is provided");
    }
}
