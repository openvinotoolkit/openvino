// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"

namespace ov::intel_gpu {

/// Classifies a Constant's role relative to the MoE fused op.
enum class MoEConstantRole {
    NotMoE,                     // Non-MoE constant (e.g. LayerNorm, embeddings, scalars)
    NotOffloadableMoE,          // MoE constant whose architecture currently lacks OTD support (e.g. GEMM2_BIAS_SWIGLU_CLAMP)
    NotOffloadableRoutedExpert, // Routed expert constant that cannot be pruned because it is shared with non-routed consumers
    OffloadableRoutedExpert,    // Pure routed expert constant eligible for partial upload and offload discounting (GEMM3_SWIGLU)
    AlwaysResidentSharedExpert  // Shared expert constant that must remain fully resident (GEMM3_SWIGLU shared inputs)
};

/// Checks if an op is an OTD-capable MOE node (MOECompressed with GEMM3_SWIGLU).
bool is_otd_capable_moe(const ov::Node* node);

/// Whether the routed weights of a MoE layer can be streamed from the weights file (if not, why, in `reason`).
/// The layers sharing routed weights with it must be eligible too.
bool is_moe_otd_eligible(const ov::Node* node, std::string* reason = nullptr);
/// Whether the constant is a routed weight of eligible MoE layers only, i.e. partially uploaded when OFFLOAD_RATIO > 0.
bool is_moe_constant_offloadable(const std::shared_ptr<ov::op::v0::Constant>& op);

/// Determines the role of a Constant feeding into MOECompressed.
MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op);

/// Number of device-resident expert slots of an offloaded (OTD) MoE layer, 0 if the layer stays fully resident (and why, in `reason`).
/// Only the routed weights are offloaded, and only if they are unmodified views of the weights file.
size_t get_moe_offload_slot_count(const ov::op::internal::MOECompressed& moe, size_t offload_ratio, std::string* reason = nullptr);

/// Offset of the data of `constant` in `weights_path`. Throws if the constant is not a view of that file.
size_t get_weights_file_offset(const ov::op::v0::Constant& constant, const std::filesystem::path& weights_path);

struct PartialUploadDesc {
    bool enabled = false;
    cldnn::memory::ptr memory = nullptr;
    ov::Shape upload_shape;
    size_t upload_bytes = 0;
};

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op);

uint64_t get_model_resident_constant_bytes(const ov::Model& model, size_t offload_ratio);
/// Throws if OFFLOAD_RATIO cannot apply to any MoE layer of the model or the resident constants exceed the device memory
void validate_model_resident_constant_memory(const ov::Model& model, size_t offload_ratio, uint64_t device_memory_bytes);

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

PartialUploadDesc try_prepare_partial_upload(cldnn::engine& engine,
                                             const ExecutionConfig& config,
                                             const std::shared_ptr<ov::op::v0::Constant>& op,
                                             const ov::Shape& const_shape,
                                             cldnn::data_types out_dtype,
                                             const cldnn::format& const_format,
                                             const cldnn::layout& const_layout);

}  // namespace ov::intel_gpu