// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "openvino/op/constant.hpp"

namespace ov::intel_gpu::moe_offload {

struct partial_upload_plan {
    cldnn::memory::ptr memory = nullptr;
    ov::Shape upload_shape;
    size_t upload_bytes = 0;
    bool skip_initial_copy = false;
};

inline size_t get_max_experts(const ExecutionConfig& config) {
    return config.get_moe_offload_max_experts();
}

inline bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    const auto users = op->get_output_target_inputs(0);
    for (const auto& input : users) {
        const auto* node = input.get_node();
        if (ov::is_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(node)) {
            return true;
        }
    }
    return false;
}

class partial_upload_log_state {
public:
    static constexpr size_t max_detailed_logs = 3;

    void log(const std::string& node_name,
             size_t uploaded_experts,
             size_t total_experts,
             size_t upload_bytes,
             size_t target_bytes) {
        total_upload_bytes.fetch_add(static_cast<uint64_t>(upload_bytes), std::memory_order_relaxed);
        total_target_bytes.fetch_add(static_cast<uint64_t>(target_bytes), std::memory_order_relaxed);

        const size_t total = total_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (total <= max_detailed_logs) {
            std::cout << "[EXPERIMENTAL] OTD partial constant allocation at compile stage: "
                      << node_name << ", experts=" << uploaded_experts
                      << "/" << total_experts << ", upload_bytes=" << upload_bytes
                      << ", target_bytes=" << target_bytes << std::endl;
        } else if (total == max_detailed_logs + 1) {
            std::cout << "[EXPERIMENTAL] OTD partial constant allocation: suppressing further per-constant logs, "
                      << "final summary will be printed at process exit" << std::endl;
        }
    }

    ~partial_upload_log_state() {
        const size_t total = total_count.load(std::memory_order_relaxed);
        if (total > max_detailed_logs) {
            const size_t shown = max_detailed_logs;
            std::cout << "[EXPERIMENTAL] OTD partial constant allocation summary: total=" << total
                      << ", shown=" << shown << ", suppressed=" << (total - shown)
                      << ", total_upload_bytes=" << total_upload_bytes.load(std::memory_order_relaxed)
                      << ", total_target_bytes=" << total_target_bytes.load(std::memory_order_relaxed)
                      << std::endl;
        }
    }

private:
    std::atomic<size_t> total_count{0};
    std::atomic<uint64_t> total_upload_bytes{0};
    std::atomic<uint64_t> total_target_bytes{0};
};

inline partial_upload_log_state& get_partial_upload_log_state() {
    static partial_upload_log_state state;
    return state;
}

inline partial_upload_plan prepare_partial_upload_plan(ProgramBuilder& p,
                                                       const std::shared_ptr<ov::op::v0::Constant>& op,
                                                       const ov::Shape& const_shape,
                                                       cldnn::data_types out_dtype,
                                                       const cldnn::format& const_format,
                                                       const cldnn::layout& const_layout) {
    partial_upload_plan plan;
    plan.upload_shape = const_shape;
    plan.upload_bytes = const_layout.bytes_count();

    const size_t otd_expert_num = get_max_experts(p.get_config());
    const bool partial_moe_const_upload = otd_expert_num > 0 && is_moe_related_constant(op);
    plan.skip_initial_copy = partial_moe_const_upload;

    if (partial_moe_const_upload && const_layout.bytes_count() > 0 && !const_shape.empty() && const_shape[0] > 0) {
        plan.upload_shape[0] = std::min<size_t>(const_shape[0], otd_expert_num);
        auto upload_layout = cldnn::layout(plan.upload_shape, out_dtype, const_format);
        auto upload_mem = p.get_engine().allocate_memory(upload_layout, false);
        plan.memory = p.get_engine().reinterpret_buffer(*upload_mem, const_layout);
        plan.upload_bytes = upload_layout.bytes_count();
        get_partial_upload_log_state().log(op->get_friendly_name(),
                                           plan.upload_shape[0],
                                           const_shape[0],
                                           plan.upload_bytes,
                                           const_layout.bytes_count());
        return plan;
    }

    if (const_layout.bytes_count() > 0) {
        plan.memory = p.get_engine().allocate_memory(const_layout, false);
        return plan;
    }

    auto one_dim_layout = cldnn::layout(ov::PartialShape({1}), const_layout.data_type, const_layout.format);
    auto one_dim_mem = p.get_engine().allocate_memory(one_dim_layout, false);
    plan.memory = p.get_engine().reinterpret_buffer(*one_dim_mem, const_layout);
    return plan;
}

inline void initialize_constant_memory(char* dst,
                                       const std::shared_ptr<ov::op::v0::Constant>& op,
                                       cldnn::data_types out_dtype,
                                       size_t upload_bytes,
                                       const ov::Shape& upload_shape,
                                       bool skip_initial_copy) {
    auto upload_count = ov::shape_size(upload_shape);
    if (skip_initial_copy) {
        return;
    }

    if (upload_count == 1 &&
        out_dtype == cldnn::data_types::f32 &&
        op->get_output_element_type(0) == ov::element::f64) {
        const auto* f64data = op->get_data_ptr<double>();
        auto f32buf = reinterpret_cast<float*>(dst);
        f32buf[0] = static_cast<float>(f64data[0]);
        return;
    }

    if (out_dtype == cldnn::data_types::f32 &&
        (op->get_output_element_type(0) == ov::element::u16 ||
         op->get_output_element_type(0) == ov::element::i16)) {
        auto f32buf = reinterpret_cast<float*>(dst);

        if (op->get_output_element_type(0) == ov::element::u16) {
            const auto* u16data = op->get_data_ptr<uint16_t>();
            for (size_t i = 0; i < upload_count; i++) {
                f32buf[i] = static_cast<float>(u16data[i]);
            }
        } else {
            const auto* i16data = op->get_data_ptr<int16_t>();
            for (size_t i = 0; i < upload_count; i++) {
                f32buf[i] = static_cast<float>(i16data[i]);
            }
        }
        return;
    }

    auto data = op->get_data_ptr<char>();
    std::memcpy(dst, data, upload_bytes);
}

}  // namespace ov::intel_gpu::moe_offload