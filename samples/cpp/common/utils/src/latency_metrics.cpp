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
    int precision = high_precision ? 6 : 2;
    
    if (high_precision) {
        // Convert to microseconds for ultra low latency applications
        stream << data_shape << ";" << std::fixed << std::setprecision(precision) 
               << (median_or_percentile * 1000.0) << ";" << (avg * 1000.0) << ";"
               << (min * 1000.0) << ";" << (max * 1000.0);
    } else {
        stream << data_shape << ";" << std::fixed << std::setprecision(precision) 
               << median_or_percentile << ";" << avg << ";"
               << min << ";" << max;
    }
    
    std::cout.flags(fmt);
}

void LatencyMetrics::write_to_slog() const {
    std::string percentileStr = (percentile_boundary == 50)
                                    ? "   Median:           "
                                    : "   " + std::to_string(percentile_boundary) + " percentile:     ";

    auto format_value = [this](double value) -> std::string {
        if (high_precision) {
            // Convert milliseconds to microseconds for ultra low latency applications
            return double_to_string_high_precision(value * 1000.0);
        } else {
            return double_to_string(value);
        }
    };
    
    std::string unit = high_precision ? " µs" : " ms";

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
