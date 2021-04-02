// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "benchmark_utils.hpp"

#ifdef PERFORMACE_BENCHMARK
/**
 * @brief use this macro to run single instruction several times by benchmark
 *
 * be carefull because macro will run only instruction to the first semicolon
 *
 * Usage:
 * ---
 * BENCHMARK_BURDEN(some_unique_name) exec_fn();
 * ---
 *
 *
 * Wrong usage:
 * ---
 * BENCHMARK_BURDEN(some_unique_name) exec_fn(); this_fn_will_not_benchmark();
 * ---
 *
 */
#define BENCHMARK_BURDEN(name) for (::perforemace_tests::Benchmark b(name); b.nextRound(); b.endRound())
#else
#define BENCHMARK_BURDEN(name)
#endif
