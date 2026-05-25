// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "samples/latency_metrics.hpp"
// clang-format on

void LatencyMetrics::write_to_stream(std::ostream& stream) const {
    std::ios::fmtflags fmt(std::cout.flags());
    stream << data_shape << ";" << std::fixed << std::setprecision(2) << median_or_percentile << ";" << avg << ";"
           << min << ";" << max;
    std::cout.flags(fmt);
}

void LatencyMetrics::write_to_slog(bool adaptive_latency_unit) const {
    std::string percentileStr = (percentile_boundary == 50)
                                    ? "   Median:           "
                                    : "   " + std::to_string(percentile_boundary) + " percentile:     ";

    const bool use_microseconds = adaptive_latency_unit && avg < 1.0;
    auto format_value = [use_microseconds](double value) -> std::string {
        return double_to_string(use_microseconds ? value * 1000.0 : value);
    };
    const char* unit = use_microseconds ? " us" : " ms";

    slog::info << percentileStr << format_value(median_or_percentile) << unit << slog::endl;
    slog::info << "   Average:          " << format_value(avg) << unit << slog::endl;
    slog::info << "   Min:              " << format_value(min) << unit << slog::endl;
    slog::info << "   Max:              " << format_value(max) << unit << slog::endl;
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
