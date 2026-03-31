// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace bench_kernel {

// ============================================================================
// High-resolution timer for kernel benchmarking
// ============================================================================

class perf_timer {
public:
    void reset() {
        times_us_.clear();
    }

    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start_).count();
        times_us_.push_back(us);
    }

    // Record a profiling time directly (from GPU event profiling, in microseconds)
    void record(double us) {
        times_us_.push_back(us);
    }

    size_t count() const { return times_us_.size(); }
    bool   empty() const { return times_us_.empty(); }

    double min_us() const {
        if (times_us_.empty()) return 0.0;
        return *std::min_element(times_us_.begin(), times_us_.end());
    }

    double max_us() const {
        if (times_us_.empty()) return 0.0;
        return *std::max_element(times_us_.begin(), times_us_.end());
    }

    double avg_us() const {
        if (times_us_.empty()) return 0.0;
        double sum = std::accumulate(times_us_.begin(), times_us_.end(), 0.0);
        return sum / static_cast<double>(times_us_.size());
    }

    double median_us() const {
        if (times_us_.empty()) return 0.0;
        std::vector<double> sorted = times_us_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        }
        return sorted[n/2];
    }

    double stddev_us() const {
        if (times_us_.size() < 2) return 0.0;
        double avg = avg_us();
        double sum_sq = 0.0;
        for (double t : times_us_) {
            sum_sq += (t - avg) * (t - avg);
        }
        return std::sqrt(sum_sq / static_cast<double>(times_us_.size() - 1));
    }

    // Print summary
    void print(const std::string& label = "", int verbose = 1) const {
        if (times_us_.empty()) {
            std::cout << label << " : no measurements" << std::endl;
            return;
        }

        std::cout << std::fixed << std::setprecision(3);
        if (!label.empty()) std::cout << label << " : ";

        std::cout << "min=" << min_us() / 1000.0 << "ms"
                  << " avg=" << avg_us() / 1000.0 << "ms"
                  << " median=" << median_us() / 1000.0 << "ms";

        if (verbose >= 2) {
            std::cout << " max=" << max_us() / 1000.0 << "ms"
                      << " stddev=" << stddev_us() / 1000.0 << "ms"
                      << " iters=" << times_us_.size();
        }
        std::cout << std::endl;
    }

    const std::vector<double>& times() const { return times_us_; }

    // Merge another timer's recorded times into this one
    void merge(const perf_timer& other) {
        for (double t : other.times_us_) {
            times_us_.push_back(t);
        }
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::vector<double> times_us_;  // all recorded times in microseconds
};

// ============================================================================
// Statistics struct for results
// ============================================================================

// ============================================================================
// Test result status enum
// ============================================================================
enum class test_status {
    passed,          // test ran and accuracy matched (or perf-only mode)
    failed,          // test ran but accuracy check failed or runtime error
    unimplemented,   // accuracy reference not implemented for this kernel/config
    skipped,         // test intentionally skipped (e.g., known slow reference)
};

inline const char* to_string(test_status s) {
    switch (s) {
        case test_status::passed:        return "PASSED";
        case test_status::failed:        return "FAILED";
        case test_status::unimplemented: return "UNIMPLEMENTED";
        case test_status::skipped:       return "SKIPPED";
        default:                         return "UNKNOWN";
    }
}

// Exception to signal unimplemented accuracy check from run_single()
struct bench_unimplemented : public std::runtime_error {
    bench_unimplemented() : std::runtime_error("accuracy check not implemented") {}
    bench_unimplemented(const std::string& msg) : std::runtime_error(msg) {}
};

// Exception to signal intentionally skipped test (e.g., known slow CPU reference)
struct bench_skipped : public std::runtime_error {
    bench_skipped() : std::runtime_error("test skipped") {}
    bench_skipped(const std::string& msg) : std::runtime_error(msg) {}
};

// Lightweight accuracy summary for aggregation (avoids circular include with bench_reference.hpp)
struct acc_summary {
    bool valid = false;          // true if accuracy was checked
    size_t total_elements = 0;
    size_t mismatches = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    bool empty() const { return !valid; }
};

struct bench_stat {
    int tests         = 0;
    int passed        = 0;
    int failed        = 0;
    int unimplemented = 0;
    int skipped       = 0;

    perf_timer perf_data;  // aggregated perf timing across all tests
    acc_summary acc_data;   // aggregated accuracy result

    void merge(const bench_stat& other) {
        tests         += other.tests;
        passed        += other.passed;
        failed        += other.failed;
        unimplemented += other.unimplemented;
        skipped       += other.skipped;
        perf_data.merge(other.perf_data);
        if (other.acc_data.valid) {
            if (!acc_data.valid) {
                acc_data = other.acc_data;
            } else {
                acc_data.total_elements += other.acc_data.total_elements;
                acc_data.mismatches += other.acc_data.mismatches;
                if (other.acc_data.max_abs_diff > acc_data.max_abs_diff)
                    acc_data.max_abs_diff = other.acc_data.max_abs_diff;
                if (other.acc_data.max_rel_diff > acc_data.max_rel_diff)
                    acc_data.max_rel_diff = other.acc_data.max_rel_diff;
            }
        }
    }

    // mode: 0=perf, 1=acc, 2=both (all use same format)
    void print(int /*mode*/ = 0) const {
        std::cout << "tests:" << tests
                  << " passed:" << passed
                  << " unimplemented:" << unimplemented
                  << " skipped:" << skipped
                  << " failed:" << failed
                  << std::endl;
        if (!perf_data.empty()) {
            std::cout << std::fixed << std::setprecision(5)
                      << "total perf: min(ms):" << perf_data.min_us() / 1000.0
                      << " avg(ms):" << perf_data.avg_us() / 1000.0
                      << std::endl;
        }
        if (!acc_data.empty()) {
            std::cout << "total acc: elements=" << acc_data.total_elements
                      << " mismatches=" << acc_data.mismatches
                      << " max_abs=" << std::scientific << std::setprecision(4) << acc_data.max_abs_diff
                      << " max_rel=" << acc_data.max_rel_diff
                      << std::endl;
        }
    }
};

}  // namespace bench_kernel
