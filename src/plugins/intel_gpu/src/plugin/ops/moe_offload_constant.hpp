// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "openvino/op/constant.hpp"

namespace ov::intel_gpu::moe_offload {

struct partial_upload_desc {
    bool enabled = false;
    cldnn::memory::ptr memory = nullptr;
    ov::Shape upload_shape;
    size_t upload_bytes = 0;
};

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
            std::cout << "MOE OTD partial constant allocation at compile stage: "
                      << node_name << ", experts=" << uploaded_experts
                      << "/" << total_experts << ", upload_bytes=" << upload_bytes
                      << ", target_bytes=" << target_bytes << std::endl;
        } else if (total == max_detailed_logs + 1) {
            std::cout << "MOE OTD partial constant allocation: suppressing further per-constant logs, "
                      << "final summary will be printed at process exit" << std::endl;
        }
    }

    ~partial_upload_log_state() {
        const size_t total = total_count.load(std::memory_order_relaxed);
        if (total > max_detailed_logs) {
            const size_t shown = max_detailed_logs;
            std::cout << "MOE OTD partial constant allocation summary: total=" << total
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

inline partial_upload_desc try_prepare_partial_upload(ProgramBuilder& p,
                                                      const std::shared_ptr<ov::op::v0::Constant>& op,
                                                      const ov::Shape& const_shape,
                                                      cldnn::data_types out_dtype,
                                                      const cldnn::format& const_format,
                                                      const cldnn::layout& const_layout) {
    partial_upload_desc desc;

    const size_t otd_expert_num = p.get_config().get_moe_offload_max_experts();
    const bool partial_moe_const_upload = otd_expert_num > 0 && is_moe_related_constant(op);
    if (!partial_moe_const_upload || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    desc.enabled = true;
    desc.upload_shape = const_shape;
    desc.upload_shape[0] = std::min<size_t>(const_shape[0], otd_expert_num);

    auto upload_layout = cldnn::layout(desc.upload_shape, out_dtype, const_format);
    auto upload_mem = p.get_engine().allocate_memory(upload_layout, false);
    desc.memory = p.get_engine().reinterpret_buffer(*upload_mem, const_layout);
    desc.upload_bytes = upload_layout.bytes_count();

    get_partial_upload_log_state().log(op->get_friendly_name(),
                                       desc.upload_shape[0],
                                       const_shape[0],
                                       desc.upload_bytes,
                                       const_layout.bytes_count());
    return desc;
}

}  // namespace ov::intel_gpu::moe_offload