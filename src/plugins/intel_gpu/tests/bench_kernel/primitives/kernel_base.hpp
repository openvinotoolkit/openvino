// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <iostream>
#include <iomanip>

#include <intel_gpu/runtime/engine.hpp>

#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"

namespace bench_kernel {

// ============================================================================
// Base class for all kernel benchmarks
// ============================================================================

class kernel_base {
public:
    virtual ~kernel_base() = default;

    // Human-readable kernel type name (e.g., "fully_connected", "gemm")
    virtual std::string name() const = 0;

    // Run benchmark for a single problem described in config
    // Returns bench_stat with pass/fail counts
    bench_stat run(cldnn::engine& engine, const bench_config& config) {
        bench_stat stat;
        stat.tests = 1;
        reported_timer_.reset();
        reported_acc_ = acc_summary{};
        try {
            run_single(engine, config);
            stat.passed = 1;
            stat.perf_data.merge(reported_timer_);
            stat.acc_data = reported_acc_;
        } catch (const bench_skipped&) {
            stat.skipped = 1;
        } catch (const bench_unimplemented&) {
            stat.unimplemented = 1;
        } catch (const std::exception& e) {
            stat.failed = 1;
            if (std::string(e.what()) != "accuracy check failed") {
                std::cout << std::fixed << std::setprecision(2);
                std::cout << config.test_index << ":" << to_string(test_status::failed)
                          << " (0.00 ms) __REPRO: " << config.repro_str() << std::endl;
            }
            std::cerr << "EXCEPTION: " << e.what() << std::endl;
        } catch (...) {
            stat.failed = 1;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << config.test_index << ":" << to_string(test_status::failed)
                      << " (0.00 ms) __REPRO: " << config.repro_str() << std::endl;
            std::cerr << "EXCEPTION: unknown" << std::endl;
        }
        return stat;
    }

protected:
    perf_timer reported_timer_;   // set by run_single() for perf aggregation
    acc_summary reported_acc_;    // set by run_single() for acc aggregation
    virtual void run_single(cldnn::engine& engine, const bench_config& config) = 0;
};

using kernel_ptr = std::shared_ptr<kernel_base>;

}  // namespace bench_kernel
