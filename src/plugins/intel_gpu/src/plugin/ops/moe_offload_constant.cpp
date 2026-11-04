// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <array>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "openvino/core/weight_sharing_util.hpp"

namespace ov::intel_gpu {

// Input indices 3..11 are routed expert weights/scales/zps (WEIGHT_0..ZP_2).
// Input indices 12..21 are shared expert weights (SHARED_GATE_WEIGHT..SHARED_GATE_GATE_WEIGHT).
// See MOE3GemmInputIndex in moe_3gemm_base.hpp for the authoritative enum.
static constexpr size_t ROUTED_INPUT_START = 3;
static constexpr size_t ROUTED_INPUT_END = 11;
static constexpr size_t SHARED_INPUT_START = 12;
static constexpr size_t SHARED_INPUT_END = 21;

MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op) {
    const auto users = op->get_output_target_inputs(0);
    for (const auto& input : users) {
        const auto* node = input.get_node();
        if (ov::is_type<ov::op::internal::MOECompressed>(node)) {
            auto idx = input.get_index();
            if (idx >= ROUTED_INPUT_START && idx <= ROUTED_INPUT_END) {
                return MoEConstantRole::RoutedExpert;
            }
            if (idx >= SHARED_INPUT_START && idx <= SHARED_INPUT_END) {
                return MoEConstantRole::SharedExpert;
            }
        }
    }
    return MoEConstantRole::NotMoE;
}

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    return get_moe_constant_role(op) != MoEConstantRole::NotMoE;
}

namespace {
using moe_input_index = cldnn::moe_3gemm_fused_compressed::input_index;
constexpr std::array<size_t, 3> routed_weight_inputs = {static_cast<size_t>(moe_input_index::weight_0),
                                                        static_cast<size_t>(moe_input_index::weight_1),
                                                        static_cast<size_t>(moe_input_index::weight_2)};

bool is_routed_weight_input(size_t index) {
    return std::find(routed_weight_inputs.begin(), routed_weight_inputs.end(), index) != routed_weight_inputs.end();
}

// Offloaded experts are streamed from the weights file as stored, so the graph must consume the file bytes unchanged:
// a constant whose data was converted, transposed or otherwise recomputed by the transformations has no file descriptor
bool is_weights_file_view(const ov::Output<ov::Node>& input, size_t num_expert) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
    if (!constant || !ov::weight_sharing::Extension::get_constant_source_buffer(*constant)) {
        return false;
    }
    const auto& shape = constant->get_shape();
    // Each expert slice must start on a byte boundary (sub-byte weights)
    return !shape.empty() && shape[0] == num_expert && (ov::shape_size(shape) * constant->get_element_type().bitwidth()) % (8 * num_expert) == 0;
}
}  // namespace

size_t get_moe_offload_slot_count(const ov::op::internal::MOECompressed& moe, const ExecutionConfig& config) {
    const size_t otd_ratio = config.get_offload_ratio();
    const auto& moe_config = moe.get_config();
    // ratio=0 (all resident) or ratio=100 (all on disk, invalid) -> no offload
    if (otd_ratio == 0 || otd_ratio >= 100 || moe_config.expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU || moe_config.num_expert == 0 ||
        moe.get_input_size() <= routed_weight_inputs.back()) {
        return 0;
    }
    const auto num_expert = static_cast<size_t>(moe_config.num_expert);
    for (const auto index : routed_weight_inputs) {
        if (!is_weights_file_view(moe.input_value(index), num_expert)) {
            return 0;
        }
    }
    // otd_ratio is the % on disk; GPU-resident experts = total * (100 - ratio) / 100
    return std::max<size_t>(1, num_expert * (100 - otd_ratio) / 100);
}

PartialUploadLogState& get_partial_upload_log_state() {
    static PartialUploadLogState state;
    return state;
}

PartialUploadDesc try_prepare_partial_upload(cldnn::engine& engine,
                                             const ExecutionConfig& config,
                                             const std::shared_ptr<ov::op::v0::Constant>& op,
                                             const ov::Shape& const_shape,
                                             cldnn::data_types out_dtype,
                                             const cldnn::format& const_format,
                                             const cldnn::layout& const_layout) {
    PartialUploadDesc desc;

    // Only the routed expert weights of an offloaded layer are partially uploaded:
    // scales, zero points and shared experts stay fully resident.
    size_t resident_expert_num = 0;
    for (const auto& input : op->get_output_target_inputs(0)) {
        const auto* moe = ov::as_type<ov::op::internal::MOECompressed>(input.get_node());
        if (moe && is_routed_weight_input(input.get_index())) {
            resident_expert_num = get_moe_offload_slot_count(*moe, config);
            break;
        }
    }
    if (resident_expert_num == 0 || const_layout.bytes_count() == 0 || const_shape.empty() || const_shape[0] == 0) {
        return desc;
    }

    desc.enabled = true;
    desc.upload_shape = const_shape;
    desc.upload_shape[0] = std::min<size_t>(const_shape[0], resident_expert_num);

    auto upload_layout = cldnn::layout(desc.upload_shape, out_dtype, const_format);
    auto upload_mem = engine.allocate_memory(upload_layout, engine.get_preferred_memory_allocation_type(), false);
    // Reinterpret the smaller physical allocation as the full constant layout so the
    // graph sees the expected shape/layout. This is safe because:
    // 1. constant.cpp marks this data node with skip_device_transfer=true (partial_upload.enabled),
    //    so no host→device memcpy of the full size occurs.
    // 2. At runtime, OTD loads on-demand into the first `resident_expert_num` slots only.
    // 3. Weightless cache serialization uses bin_offset metadata for these constants and
    //    never reads the buffer contents via mem->buffer_ptr(). OTD provides weights_path,
    //    but does not enable weightless caching itself.
    // TODO: Support serialization of OTD partial allocations without weightless caching.
    OPENVINO_ASSERT(upload_layout.bytes_count() <= const_layout.bytes_count(),
                    "Partial upload layout (", upload_layout.bytes_count(),
                    " bytes) exceeds full constant layout (", const_layout.bytes_count(), " bytes)");
    desc.memory = engine.reinterpret_buffer(*upload_mem, const_layout);
    desc.upload_bytes = upload_layout.bytes_count();

    get_partial_upload_log_state().log(op->get_friendly_name(),
                                       desc.upload_shape[0],
                                       const_shape[0],
                                       desc.upload_bytes,
                                       const_layout.bytes_count());
    return desc;
}

}  // namespace ov::intel_gpu
