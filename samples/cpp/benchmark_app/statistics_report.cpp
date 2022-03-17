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

static const char* status_names[] = {"NOT_RUN", "OPTIMIZED_OUT", "EXECUTED"};

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
        dumper << ((int)layer.status < (sizeof(status_names) / sizeof(status_names[0]))
                       ? status_names[(int)layer.status]
                       : "INVALID_STATUS");
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

StatisticsReport::PerformanceCounters StatisticsReport::get_average_performance_counters(
    const std::vector<PerformanceCounters>& perfCounts) {
    StatisticsReport::PerformanceCounters performanceCountersAvg;
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
    CsvDumper dumper(true, _config.report_folder + _separator + "benchmark_" + _config.report_type + "_report.csv");
    if (_config.report_type == detailedCntReport) {
        for (auto& pc : perfCounts) {
            dump_performance_counters_request(dumper, pc);
        }
    } else if (_config.report_type == averageCntReport) {
        dump_performance_counters_request(dumper, get_average_performance_counters(perfCounts));
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
            ((int)layer.status < (sizeof(status_names) / sizeof(status_names[0])) ? status_names[(int)layer.status]
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

void LatencyMetrics::write_to_stream(std::ostream& stream) const {
    std::ios::fmtflags fmt(std::cout.flags());
    stream << data_shape << ";" << std::fixed << std::setprecision(2) << median_or_percentile << ";" << avg << ";"
           << min << ";" << max;
    std::cout.flags(fmt);
}

void LatencyMetrics::write_to_slog() const {
    std::string percentileStr = (percentile_boundary == 50)
                                    ? "\tMedian:     "
                                    : "\t" + std::to_string(percentile_boundary) + " percentile:    ";
    if (!data_shape.empty()) {
        slog::info << "\tData shape: " << data_shape << slog::endl;
    }
    slog::info << percentileStr << double_to_string(median_or_percentile) << " ms" << slog::endl;
    slog::info << "\tAverage:    " << double_to_string(avg) << " ms" << slog::endl;
    slog::info << "\tMin:        " << double_to_string(min) << " ms" << slog::endl;
    slog::info << "\tMax:        " << double_to_string(max) << " ms" << slog::endl;
}

const nlohmann::json LatencyMetrics::to_json() const {
    nlohmann::json stat;
    stat["data_shape"] = data_shape;
    stat["latency_median"] = median_or_percentile;
    stat["latency_average"] = avg;
    stat["latency_min"] = min;
    stat["latency_max"] = max;
    return stat;
}

void LatencyMetrics::fill_data(std::vector<double> latencies, size_t percentile_boundary) {
    if (latencies.empty()) {
        throw std::logic_error("Latency metrics class expects non-empty vector of latencies at consturction.");
    }
    std::sort(latencies.begin(), latencies.end());
    min = latencies[0];
    avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    median_or_percentile = latencies[size_t(latencies.size() / 100.0 * percentile_boundary)];
    max = latencies.back();
};

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
        arr.push_back(metrics_val.to_json());
    } break;
    default:
        throw std::invalid_argument("StatisticsVariant:: json conversion : invalid type is provided");
    }
}
