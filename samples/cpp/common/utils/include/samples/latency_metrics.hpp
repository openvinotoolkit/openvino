// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "samples/common.hpp"
#include "samples/slog.hpp"
// clang-format on

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

    double median_or_percentile = 0;
    double avg = 0;
    double min = 0;
    double max = 0;
    std::string data_shape;

private:
    void fill_data(std::vector<double> latencies, size_t percentile_boundary);
    size_t percentile_boundary = 50;
};
