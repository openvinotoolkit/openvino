// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>
#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "openvino/core/type/float16.hpp"

#include "bench_types.hpp"
#include "bench_attrs.hpp"
#include "bench_config.hpp"
#include "bench_timer.hpp"

namespace bench_kernel {

// ============================================================================
// Memory fill helpers
// ============================================================================

template <typename T>
static void fill_random(cldnn::memory::ptr mem, cldnn::stream& stream, float low = -1.0f, float high = 1.0f) {
    auto lock = cldnn::mem_lock<T>(mem, stream);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < lock.size(); ++i) {
        lock[i] = static_cast<T>(dist(gen));
    }
}

inline void fill_memory_random(cldnn::memory::ptr mem, cldnn::stream& stream, cldnn::data_types dt) {
    switch (dt) {
        case cldnn::data_types::f32:
            fill_random<float>(mem, stream);
            break;
        case cldnn::data_types::f16:
            fill_random<ov::float16>(mem, stream);
            break;
        case cldnn::data_types::i8:
            fill_random<int8_t>(mem, stream, -5.0f, 5.0f);
            break;
        case cldnn::data_types::u8:
            fill_random<uint8_t>(mem, stream, 0.0f, 10.0f);
            break;
        case cldnn::data_types::i32:
            fill_random<int32_t>(mem, stream, -100.0f, 100.0f);
            break;
        case cldnn::data_types::i64:
            fill_random<int64_t>(mem, stream, -100.0f, 100.0f);
            break;
        default:
            // For i4/u4 etc., fill underlying bytes
            {
                auto lock = cldnn::mem_lock<uint8_t>(mem, stream);
                std::mt19937 gen(42);
                std::uniform_int_distribution<int> dist(0, 255);
                for (size_t i = 0; i < lock.size(); ++i) {
                    lock[i] = static_cast<uint8_t>(dist(gen));
                }
            }
            break;
    }
}

// ============================================================================
// Format selection based on tensor rank
// ============================================================================

inline cldnn::format get_format_for_rank(size_t rank) {
    switch (rank) {
        case 1:
        case 2:
        case 3:
        case 4:  return cldnn::format::bfyx;
        case 5:  return cldnn::format::bfzyx;
        case 6:  return cldnn::format::bfwzyx;
        default: return cldnn::format::bfyx;
    }
}

// ============================================================================
// String-to-format conversion (reverse of format::to_string())
// ============================================================================

inline cldnn::format str_to_format(const std::string& s) {
    if (s.empty() || s == "any") return cldnn::format::any;
    for (int i = 0; i < static_cast<int>(cldnn::format::format_num); ++i) {
        auto ft = static_cast<cldnn::format::type>(i);
        try {
            if (cldnn::format(ft).to_string() == s) return ft;
        } catch (...) {}
    }
    return cldnn::format::any;  // unrecognized → default
}

// Parse comma-separated format list (e.g., "bfyx,b_fs_yx_fsv16" → vector)
inline std::vector<cldnn::format> parse_format_list(const std::string& s) {
    std::vector<cldnn::format> fmts;
    if (s.empty()) return fmts;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        fmts.push_back(str_to_format(tok));
    }
    return fmts;
}

// Get input format for a given input index from config, falling back to rank-based default
inline cldnn::format get_input_format(const bench_config& config, size_t idx, size_t rank) {
    // Accuracy/reference path assumes plain logical tensor order when reading
    // input data for CPU reference computation. For blocked GPU input formats
    // (e.g., b_fs_yx_fsv16), direct input allocation in blocked layout can make
    // GPU-vs-ref comparison diverge due to layout interpretation differences.
    // Keep input allocation in default rank-based format for correctness mode.
    if (config.is_acc()) return get_format_for_rank(rank);

    if (config.in_layouts_str.empty()) return get_format_for_rank(rank);
    auto fmts = parse_format_list(config.in_layouts_str);
    if (idx < fmts.size() && fmts[idx] != cldnn::format::any) return fmts[idx];
    return get_format_for_rank(rank);
}

// Get output format from config (for ImplementationDesc)
inline cldnn::format get_output_format(const bench_config& config) {
    // Same rationale as get_input_format(): correctness mode prioritizes
    // stable GPU-vs-reference comparison over blocked-format forcing.
    if (config.is_acc()) return cldnn::format::any;

    if (config.out_layouts_str.empty()) return cldnn::format::any;
    auto fmts = parse_format_list(config.out_layouts_str);
    if (!fmts.empty() && fmts[0] != cldnn::format::any) return fmts[0];
    return cldnn::format::any;
}

inline cldnn::data_types get_terminal_post_op_dtype(const std::vector<post_op_entry>& post_ops,
                                                    cldnn::data_types output_dt) {
    if (!post_ops.empty() && post_ops.back().kind == post_op_kind::quantize)
        return post_ops.back().quant_out_dt;
    return output_dt;
}

inline void add_terminal_post_op_consumer_reorder(cldnn::topology& topology,
                                                  const bench_config& config,
                                                  const std::vector<post_op_entry>& post_ops,
                                                  std::string& last_prim_id,
                                                  cldnn::data_types output_dt) {
    if (post_ops.empty() || config.is_acc())
        return;

    std::string consumer_id = last_prim_id + "_consumer";
    cldnn::format consumer_fmt = get_output_format(config);
    cldnn::data_types consumer_dt = get_terminal_post_op_dtype(post_ops, output_dt);
    topology.add(cldnn::reorder(consumer_id,
                                cldnn::input_info(last_prim_id),
                                consumer_fmt,
                                consumer_dt));
    last_prim_id = consumer_id;
}

// ============================================================================
// Convert grouped weight shape to standard format for reference computation
// Grouped:  [G, OC/G, IC/G, spatial...]  (rank = 2 + 1 + spatial_dims)
// Standard: [OC, IC/G, spatial...]        (rank = 2 + spatial_dims)
// The flat memory layout is identical, only shape interpretation changes.
// ============================================================================

inline std::vector<int64_t> convert_grouped_weight_shape(const std::vector<int64_t>& grouped_shape) {
    if (grouped_shape.size() < 3) return grouped_shape;
    std::vector<int64_t> std_shape;
    std_shape.push_back(grouped_shape[0] * grouped_shape[1]);  // OC = G * OC/G
    for (size_t i = 2; i < grouped_shape.size(); ++i)
        std_shape.push_back(grouped_shape[i]);
    return std_shape;
}

// ============================================================================
// Map bench_kernel activation_func to cldnn::activation_func
// ============================================================================

inline cldnn::activation_func map_activation(activation_func f, float alpha = 0.0f) {
    switch (f) {
        case activation_func::relu:
            // relu with non-zero alpha is LeakyReLU (relu_negative_slope)
            return (alpha != 0.0f) ? cldnn::activation_func::relu_negative_slope
                                   : cldnn::activation_func::relu;
        case activation_func::sigmoid:     return cldnn::activation_func::logistic;
        case activation_func::tanh:        return cldnn::activation_func::hyperbolic_tan;
        case activation_func::elu:         return cldnn::activation_func::elu;
        case activation_func::abs:         return cldnn::activation_func::abs;
        case activation_func::sqrt:        return cldnn::activation_func::sqrt;
        case activation_func::square:      return cldnn::activation_func::square;
        case activation_func::exp:         return cldnn::activation_func::exp;
        case activation_func::log:         return cldnn::activation_func::log;
        case activation_func::gelu_erf:    return cldnn::activation_func::gelu;
        case activation_func::gelu_tanh:   return cldnn::activation_func::gelu_tanh;
        case activation_func::swish:       return cldnn::activation_func::swish;
        case activation_func::mish:        return cldnn::activation_func::mish;
        case activation_func::hardswish:   return cldnn::activation_func::hswish;
        case activation_func::hardsigmoid: return cldnn::activation_func::hard_sigmoid;
        case activation_func::softplus:    return cldnn::activation_func::softplus;
        case activation_func::clip:        return cldnn::activation_func::clamp;
        case activation_func::clamp:       return cldnn::activation_func::clamp;
        case activation_func::round:       return cldnn::activation_func::round_half_to_even;
        case activation_func::linear:      return cldnn::activation_func::linear;
    }
    return cldnn::activation_func::relu;
}

// ============================================================================
// Map bench_kernel eltwise_mode to cldnn::eltwise_mode
// ============================================================================

inline cldnn::eltwise_mode map_eltwise_mode(eltwise_mode m) {
    switch (m) {
        case eltwise_mode::sum:      return cldnn::eltwise_mode::sum;
        case eltwise_mode::prod:     return cldnn::eltwise_mode::prod;
        case eltwise_mode::sub:      return cldnn::eltwise_mode::sub;
        case eltwise_mode::div_mode: return cldnn::eltwise_mode::div;
    }
    return cldnn::eltwise_mode::sum;
}

// ============================================================================
// Profiling time extraction
// ============================================================================

inline double get_exec_time_us(const std::map<cldnn::primitive_id, cldnn::network_output>& outputs,
                                const std::string& prim_id) {
    auto it = outputs.find(prim_id);
    if (it == outputs.end()) return 0.0;

    auto event = it->second.get_event();
    if (!event) return 0.0;
    event->wait();

    auto intervals = event->get_profiling_info();
    for (const auto& interval : intervals) {
        if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
            return std::chrono::duration<double, std::micro>(interval.value->value()).count();
        }
    }
    return 0.0;
}

// ============================================================================
// Total network profiling time (sum of all primitives' executing stage)
// ============================================================================

inline double get_total_exec_time_us(const std::map<cldnn::primitive_id, cldnn::network_output>& outputs) {
    double total_us = 0.0;
    for (const auto& [id, output] : outputs) {
        auto event = output.get_event();
        if (!event) continue;
        event->wait();
        for (const auto& interval : event->get_profiling_info()) {
            if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                total_us += std::chrono::duration<double, std::micro>(interval.value->value()).count();
            }
        }
    }
    return total_us;
}

// ============================================================================
// ExecutionConfig helper — creates config with profiling and optional impl forcing
// ============================================================================

inline cldnn::ExecutionConfig make_exec_config(const bench_config& config,
                                                const std::string& prim_id = "") {
    cldnn::ExecutionConfig exec_config;
    exec_config.set_property(ov::enable_profiling(true));
    exec_config.set_property(ov::intel_gpu::optimize_data(true));
    exec_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if ((config.impl != impl_type::any || !config.force_impl_str.empty() || !config.out_layouts_str.empty()) && !prim_id.empty()) {
        // Use output format from config (from BenchVerbose log) if available,
        // which determines kernel selection (e.g., opt vs ref kernel).
        // Falls back to format::any (runtime chooses) when not specified.
        cldnn::format out_fmt = get_output_format(config);
        ov::intel_gpu::ImplementationDesc impl_desc = {out_fmt, config.force_impl_str, config.impl};
        exec_config.set_property(
            ov::intel_gpu::force_implementations(
                ov::intel_gpu::ImplForcingMap{{prim_id, impl_desc}}));
    }
    return exec_config;
}

// ============================================================================
// Performance measurement loop helper
// ============================================================================

inline void run_perf(cldnn::network& network, const bench_config& config, perf_timer& timer) {
    for (int i = 0; i < config.warmup_iters; ++i)
        network.execute();
    for (int i = 0; i < config.perf_iters; ++i) {
        network.execute();
        auto executed = network.get_executed_primitives();
        double gpu_time = 0.0;
        for (const auto& [id, event] : executed) {
            if (!event) continue;
            event->wait();
            for (const auto& interval : event->get_profiling_info()) {
                if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                    gpu_time += std::chrono::duration<double, std::micro>(interval.value->value()).count();
                }
            }
        }
        if (gpu_time > 0) timer.record(gpu_time);
        else { timer.start(); timer.stop(); }
    }
}

// ============================================================================
// Result printing and finalization helpers
// ============================================================================

// Overload without acc/timer (for kernels that don't need verbose details)
inline void print_result(const cldnn::network& network,
                         const std::string& impl_info_prim_id,
                         const bench_config& config,
                         bool test_passed,
                         bool acc_unimplemented,
                         double wall_ms) {
    std::cout << "impl_info: " << network.get_implementation_info(impl_info_prim_id) << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << config.test_index << ":";
    if (acc_unimplemented) std::cout << to_string(test_status::unimplemented);
    else if (test_passed) std::cout << to_string(test_status::passed);
    else std::cout << to_string(test_status::failed);
    std::cout << " (" << wall_ms << " ms)";
    std::cout << " __REPRO: " << config.repro_str() << std::endl;
}

// Full overload with acc and timer (AccResult is deduced from caller, e.g. acc_result)
template <typename AccResult>
inline void print_result(const cldnn::network& network,
                         const std::string& impl_info_prim_id,
                         const bench_config& config,
                         bool test_passed,
                         bool acc_unimplemented,
                         double wall_ms,
                         const AccResult* acc_res,
                         const perf_timer* timer) {
    std::cout << "impl_info: " << network.get_implementation_info(impl_info_prim_id) << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << config.test_index << ":";
    if (acc_unimplemented) std::cout << to_string(test_status::unimplemented);
    else if (test_passed) std::cout << to_string(test_status::passed);
    else std::cout << to_string(test_status::failed);
    std::cout << " (" << wall_ms << " ms)";
    std::cout << " __REPRO: " << config.repro_str() << std::endl;
    if (config.verbose >= 2) {
        if (acc_res) { std::cout << "  "; acc_res->print("", config.verbose); }
        if (timer)   { std::cout << "  "; timer->print("", config.verbose); }
    }
}

// Overload for callers passing nullptr, nullptr (exec-only kernels without accuracy)
inline void print_result(const cldnn::network& network,
                         const std::string& impl_info_prim_id,
                         const bench_config& config,
                         bool test_passed,
                         bool acc_unimplemented,
                         double wall_ms,
                         std::nullptr_t,
                         std::nullptr_t) {
    print_result(network, impl_info_prim_id, config, test_passed, acc_unimplemented, wall_ms);
}

// Overload for callers passing nullptr for acc_res but valid timer pointer
inline void print_result(const cldnn::network& network,
                         const std::string& impl_info_prim_id,
                         const bench_config& config,
                         bool test_passed,
                         bool acc_unimplemented,
                         double wall_ms,
                         std::nullptr_t,
                         const perf_timer* timer) {
    print_result(network, impl_info_prim_id, config, test_passed, acc_unimplemented, wall_ms);
    if (config.verbose >= 2 && timer) {
        std::cout << "  "; timer->print("", config.verbose);
    }
}

inline void finalize_result(bool test_passed, bool acc_unimplemented = false) {
    if (acc_unimplemented)
        throw bench_unimplemented("post-op config not supported for CPU reference");
    if (!test_passed)
        throw std::runtime_error("accuracy check failed");
}

// Print skip result without requiring a network object (skips happen before network build)
inline void print_skip_result(const bench_config& config, const std::string& reason) {
    std::cout << "impl_info: skipped" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << config.test_index << ":" << to_string(test_status::skipped)
              << " (0.00 ms) " << reason
              << " __REPRO: " << config.repro_str() << std::endl;
}

}  // namespace bench_kernel
