// Copyright (C) 2018-2025 Intel Corporation
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

void LatencyMetrics::write_to_slog() const {
    std::string percentileStr = (percentile_boundary == 50)
                                    ? "   Median:           "
                                    : "   " + std::to_string(percentile_boundary) + " percentile:     ";

    slog::info << percentileStr << double_to_string(median_or_percentile) << " ms" << slog::endl;
    slog::info << "   Average:          " << double_to_string(avg) << " ms" << slog::endl;
    slog::info << "   Min:              " << double_to_string(min) << " ms" << slog::endl;
    slog::info << "   Max:              " << double_to_string(max) << " ms" << slog::endl;
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
