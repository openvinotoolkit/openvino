// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/op/constant.hpp"

namespace ov::intel_gpu::moe_offload {

/// Classifies a Constant's role relative to the MoE fused op.
enum class MoEConstantRole { NotMoE, RoutedExpert, SharedExpert };

/// Determines the role of a Constant feeding into MOECompressed.
MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op);

struct PartialUploadDesc {
    bool enabled = false;
    cldnn::memory::ptr memory = nullptr;
    ov::Shape upload_shape;
    size_t upload_bytes = 0;
};

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op);

class PartialUploadLogState {
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
            GPU_DEBUG_INFO << "MOE OTD partial constant allocation at compile stage: "
                           << node_name << ", experts=" << uploaded_experts
                           << "/" << total_experts << ", upload_bytes=" << upload_bytes
                           << ", target_bytes=" << target_bytes << std::endl;
        } else if (total == max_detailed_logs + 1) {
            GPU_DEBUG_INFO << "MOE OTD partial constant allocation: suppressing further per-constant logs, "
                           << "final summary will be printed at process exit" << std::endl;
        }
    }

    ~PartialUploadLogState() {
        const size_t total = total_count.load(std::memory_order_relaxed);
        if (total > max_detailed_logs) {
            const size_t shown = max_detailed_logs;
            GPU_DEBUG_INFO << "MOE OTD partial constant allocation summary: total=" << total
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

PartialUploadLogState& get_partial_upload_log_state();

PartialUploadDesc try_prepare_partial_upload(ProgramBuilder& p,
                                             const std::shared_ptr<ov::op::v0::Constant>& op,
                                             const ov::Shape& const_shape,
                                             cldnn::data_types out_dtype,
                                             const cldnn::format& const_format,
                                             const cldnn::layout& const_layout);

}  // namespace ov::intel_gpu::moe_offload