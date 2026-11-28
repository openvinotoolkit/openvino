// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <deque>
#include <optional>

#include "../common.hpp"

namespace ov {
namespace auto_plugin {

/**
 * @brief Measures how often a backend refreshes its data, from the times at which a query saw new data.
 *
 * The period is the median of the gaps between refreshes. Only gaps between two precise detections count, i.e. the
 * query before each was close enough in time to tell when the refresh happened. A gap that is a whole number of
 * periods (a refresh was skipped on purpose) is divided down. The period is only reported while the gaps agree, so
 * a backend that does not refresh regularly never gets one.
 */
class PeriodEstimator {
public:
    struct Config {
        size_t max_samples = 9;
        size_t min_samples = 3;
        std::chrono::milliseconds min_period{20};
        std::chrono::milliseconds max_period{10000};
        // mean absolute deviation, relative to the median, above which the gaps are not regular
        double max_spread = 0.5;
        // how far a gap divided by a whole number may be from the known period
        double tolerance = 0.2;
        // consecutive gaps off the known period after which the samples are dropped and measured again
        size_t max_rejected = 5;
    };

    PeriodEstimator() : PeriodEstimator(Config{}) {}
    explicit PeriodEstimator(Config config) : m_config(config) {}

    // a query at `at` saw new data; `precise`: the query before it was close enough to pin down the refresh;
    // `max_skipped`: how many refreshes the queries may have skipped on purpose since the last detection
    void on_change(Time at, bool precise, unsigned max_skipped = 1);
    // no new data came for far longer than the period
    void on_lost();
    void reset();

    std::optional<std::chrono::milliseconds> period() const;
    bool reliable() const;
    bool lost() const {
        return m_lost;
    }

private:
    std::optional<double> median_ms() const;

    Config m_config;
    std::deque<double> m_samples;
    std::optional<Time> m_last;
    bool m_last_precise = false;
    bool m_lost = false;
    size_t m_rejected = 0;
};

}  // namespace auto_plugin
}  // namespace ov
