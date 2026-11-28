// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_offload_constant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <dxgi1_4.h>
#    include <wrl/client.h>
#    pragma comment(lib, "dxgi.lib")
#endif
namespace ov::intel_gpu {

// Input indices 3..11 are routed expert weights/scales/zps (WEIGHT_0..ZP_2).
// Input indices 12..21 are shared expert weights (SHARED_GATE_WEIGHT..SHARED_GATE_GATE_WEIGHT).
// See MOE3GemmInputIndex in moe_3gemm_base.hpp for the authoritative enum.
static constexpr size_t ROUTED_INPUT_START = 3;
static constexpr size_t ROUTED_INPUT_END = 11;
static constexpr size_t SHARED_INPUT_START = 12;
static constexpr size_t SHARED_INPUT_END = 21;
static constexpr double AUTO_OFFLOAD_RATIO_FIT_SAFETY = 0.85;

bool is_otd_capable_moe(const ov::Node* node) {
    if (const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(node)) {
        return moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    }
    return false;
}

MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op) {
    if (!op) {
        return MoEConstantRole::NotMoE;
    }

    const auto target_inputs = op->get_output_target_inputs(0);
    bool has_moe_consumer = false;
    bool has_non_otd_moe = false;
    bool has_routed = false;
    bool has_shared = false;
    bool has_other_consumer = false;

    for (const auto& input : target_inputs) {
        const auto* node = input.get_node();
        if (const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(node)) {
            has_moe_consumer = true;
            if (moe->get_config().expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
                has_non_otd_moe = true;
            } else {
                const auto idx = input.get_index();
                if (idx >= ROUTED_INPUT_START && idx <= ROUTED_INPUT_END) {
                    has_routed = true;
                } else if (idx >= SHARED_INPUT_START && idx <= SHARED_INPUT_END) {
                    has_shared = true;
                } else {
                    has_other_consumer = true;
                }
            }
        } else {
            has_other_consumer = true;
        }
    }

    if (!has_moe_consumer) {
        return MoEConstantRole::NotMoE;
    }
    if (has_non_otd_moe) {
        return MoEConstantRole::NotOffloadableMoE;
    }
    if (has_shared) {
        return MoEConstantRole::AlwaysResidentSharedExpert;
    }
    if (has_routed) {
        if (has_other_consumer) {
            return MoEConstantRole::NotOffloadableRoutedExpert;
        }
        return MoEConstantRole::OffloadableRoutedExpert;
    }
    return MoEConstantRole::NotOffloadableMoE;
}

namespace {
using moe_input_index = cldnn::moe_3gemm_fused_compressed::input_index;
constexpr std::array<size_t, 3> routed_weight_inputs = {static_cast<size_t>(moe_input_index::weight_0),
                                                        static_cast<size_t>(moe_input_index::weight_1),
                                                        static_cast<size_t>(moe_input_index::weight_2)};
constexpr std::array<size_t, 6> routed_param_inputs = {static_cast<size_t>(moe_input_index::scale_0),
                                                       static_cast<size_t>(moe_input_index::zp_0),
                                                       static_cast<size_t>(moe_input_index::scale_1),
                                                       static_cast<size_t>(moe_input_index::zp_1),
                                                       static_cast<size_t>(moe_input_index::scale_2),
                                                       static_cast<size_t>(moe_input_index::zp_2)};

bool is_routed_weight_input(size_t index) {
    return std::find(routed_weight_inputs.begin(), routed_weight_inputs.end(), index) != routed_weight_inputs.end();
}

bool is_routed_param_input(size_t index) {
    return std::find(routed_param_inputs.begin(), routed_param_inputs.end(), index) != routed_param_inputs.end();
}

// Offloaded experts are streamed from the weights file as stored, so the graph must consume the file bytes unchanged:
// a constant whose data was converted, transposed or otherwise recomputed by the transformations has no file descriptor
bool is_weights_file_view(const ov::op::v0::Constant& constant, size_t num_expert) {
    if (!ov::weight_sharing::Extension::get_constant_source_buffer(constant)) {
        return false;
    }
    const auto& shape = constant.get_shape();
    // Each expert slice must start on a byte boundary (sub-byte weights)
    return !shape.empty() && shape[0] == num_expert && (ov::shape_size(shape) * constant.get_element_type().bitwidth()) % (8 * num_expert) == 0;
}

// The slots of an offloaded layer copy a scale / zero point per expert, unless it is shared by all the experts (at most one element)
bool is_shared_or_sliced_per_expert(const ov::Output<ov::Node>& input, size_t num_expert) {
    if (input.get_partial_shape().is_dynamic()) {
        return false;
    }
    const auto& shape = input.get_shape();
    const auto count = ov::shape_size(shape);
    return count <= 1 || (shape[0] == num_expert && (count * input.get_element_type().bitwidth()) % (8 * num_expert) == 0);
}

// Why the layer must stay fully resident, nullptr if it can be offloaded. Collects the layers that share its routed weights.
const char* get_resident_reason(const ov::Node* node, std::vector<const ov::Node*>& sharing_layers) {
    const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(node);
    if (!moe || !is_otd_capable_moe(moe) || moe->get_config().num_expert == 0 || moe->get_input_size() <= routed_param_inputs.back()) {
        return "the expert type is not supported";
    }
    const auto num_expert = static_cast<size_t>(moe->get_config().num_expert);
    for (const auto index : routed_weight_inputs) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(moe->get_input_node_shared_ptr(index));
        if (!constant || get_moe_constant_role(constant) != MoEConstantRole::OffloadableRoutedExpert) {
            return "the routed weights are not constants used by MoE layers only";
        }
        for (const auto& consumer : constant->get_output_target_inputs(0)) {
            if (!is_routed_weight_input(consumer.get_index())) {
                return "the routed weights are not constants used by MoE layers only";
            }
            sharing_layers.push_back(consumer.get_node());
        }
        if (!is_weights_file_view(*constant, num_expert)) {
            return "the routed weights are not unmodified views of the weights file";
        }
    }
    // Only the routed weights are streamed: the scales and zero points stay resident, with per-expert slot copies
    for (const auto index : routed_param_inputs) {
        if (!is_shared_or_sliced_per_expert(moe->input_value(index), num_expert)) {
            return "a scale or zero point is neither shared by all the experts nor sliced per expert";
        }
    }
    return nullptr;
}
}  // namespace

bool is_moe_otd_eligible(const ov::Node* node, std::string* reason) {
    // Layers sharing a routed weight read the same partially uploaded constant: they are offloaded together or not at all
    std::vector<const ov::Node*> pending{node};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        const auto* current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        if (const auto* why = get_resident_reason(current, pending)) {
            if (reason) {
                *reason = why;
            }
            return false;
        }
    }
    return true;
}

bool is_moe_constant_offloadable(const std::shared_ptr<ov::op::v0::Constant>& op) {
    if (get_moe_constant_role(op) != MoEConstantRole::OffloadableRoutedExpert) {
        return false;
    }
    const auto consumers = op->get_output_target_inputs(0);
    return std::all_of(consumers.begin(), consumers.end(), [](const ov::Input<ov::Node>& consumer) {
        return is_routed_weight_input(consumer.get_index()) && is_moe_otd_eligible(consumer.get_node());
    });
}

bool is_moe_related_constant(const std::shared_ptr<ov::op::v0::Constant>& op) {
    return get_moe_constant_role(op) != MoEConstantRole::NotMoE;
}

namespace {
// Slot count of the offloaded layers that consume the constant as a routed weight, 0 for any other constant
size_t get_routed_weight_slot_count(const std::shared_ptr<ov::op::v0::Constant>& op, size_t offload_ratio) {
    if (offload_ratio == 0 || !is_moe_constant_offloadable(op)) {
        return 0;
    }
    // Its consumers are offloaded together, with the same number of experts
    const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(op->get_output_target_inputs(0).begin()->get_node());
    return get_moe_offload_slot_count(*moe, offload_ratio);
}

// Bytes of the slot copies of a per-expert scale / zero point, one set per offloaded layer that consumes it
uint64_t get_routed_param_slot_bytes(const std::shared_ptr<ov::op::v0::Constant>& op, size_t offload_ratio) {
    const auto& shape = op->get_shape();
    if (ov::shape_size(shape) <= 1) {
        return 0;
    }
    uint64_t bytes = 0;
    for (const auto& input : op->get_output_target_inputs(0)) {
        const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(input.get_node());
        if (!moe || !is_routed_param_input(input.get_index())) {
            continue;
        }
        const auto slot_count = get_moe_offload_slot_count(*moe, offload_ratio);
        if (slot_count > 0 && shape[0] > 0) {
            bytes += op->get_byte_size() / shape[0] * std::min<size_t>(shape[0], slot_count);
        }
    }
    return bytes;
}
}  // namespace

size_t get_moe_offload_slot_count(const ov::op::internal::MOECompressed& moe, size_t offload_ratio, std::string* reason) {
    const auto resident = [reason](const char* why) -> size_t {
        if (reason) {
            *reason = why;
        }
        return 0;
    };
    // ratio=0 keeps all experts resident
    if (offload_ratio == 0) {
        return resident("OFFLOAD_RATIO is 0");
    }
    if (!is_moe_otd_eligible(&moe, reason)) {
        return 0;
    }
    // offload_ratio is the % on disk; GPU-resident experts = max(1, total * (100 - ratio) / 100)
    return std::max<size_t>(1, static_cast<size_t>(moe.get_config().num_expert) * (100 - offload_ratio) / 100);
}

size_t get_weights_file_offset(const ov::op::v0::Constant& constant, const std::filesystem::path& weights_path) {
    std::ifstream weights_file(weights_path, std::ios::binary);
    OPENVINO_ASSERT(weights_file.is_open(), "OTD weights file cannot be opened: ", weights_path);
    const auto file_size = static_cast<size_t>(std::filesystem::file_size(weights_path));
    const auto source = ov::weight_sharing::Extension::get_constant_source_buffer(constant);
    const auto byte_size = constant.get_byte_size();
    // With mmap the descriptor offset is relative to the whole file. Without it, it is relative to a buffer read for one constant,
    // whose file range is in the IR attribute.
    auto offset = static_cast<size_t>(ov::weight_sharing::Extension::get_constant_id(constant));
    if (!source || source->size() != file_size) {
        const auto origin = ov::weight_sharing::Extension::get_constant_origin(constant);
        OPENVINO_ASSERT(source && origin && origin->m_size == byte_size,
                        "OTD: constant ",
                        constant.get_friendly_name(),
                        " cannot be located in ",
                        weights_path);
        offset = origin->m_offset;
    }
    OPENVINO_ASSERT(offset <= file_size && byte_size <= file_size - offset,
                    "OTD: constant ",
                    constant.get_friendly_name(),
                    " at offset ",
                    offset,
                    " (",
                    byte_size,
                    " bytes) is out of ",
                    weights_path,
                    " (",
                    file_size,
                    " bytes)");
    // The slices are streamed from this file: compare its first and last bytes with the constant
    const auto* data = static_cast<const char*>(constant.get_data_ptr());
    const size_t sample_size = std::min<size_t>(byte_size, 64);
    for (const size_t sample_offset : {size_t{0}, byte_size - sample_size}) {
        std::array<char, 64> sample{};
        weights_file.seekg(static_cast<std::streamoff>(offset + sample_offset));
        weights_file.read(sample.data(), static_cast<std::streamsize>(sample_size));
        OPENVINO_ASSERT(weights_file && std::equal(sample.begin(), sample.begin() + sample_size, data + sample_offset),
                        "OTD: ",
                        weights_path,
                        " does not contain the data of constant ",
                        constant.get_friendly_name(),
                        ": ov::weights_path must be the weights file the model was read from");
    }
    return offset;
}

uint64_t get_model_resident_constant_bytes(const ov::Model& model, size_t offload_ratio) {
    OPENVINO_ASSERT(offload_ratio <= 100, "OFFLOAD_RATIO must be in the range [0, 100]");

    uint64_t resident_bytes = 0;
    bool has_moe = false;
    std::unordered_set<const ov::Node*> visited;
    std::function<void(const ov::Model&)> collect_resident_bytes = [&](const ov::Model& current_model) {
        for (const auto& node : current_model.get_ops()) {
            has_moe = has_moe || is_otd_capable_moe(node.get());
            if (auto subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                for (const auto& subgraph : subgraph_op->get_functions()) {
                    collect_resident_bytes(*subgraph);
                }
            }

            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!constant || !visited.insert(constant.get()).second) {
                continue;
            }

            const auto role = get_moe_constant_role(constant);
            if (role == MoEConstantRole::NotMoE) {
                continue;
            }

            uint64_t constant_resident_bytes = constant->get_byte_size();
            if (const size_t slot_count = get_routed_weight_slot_count(constant, offload_ratio)) {
                const auto& shape = constant->get_shape();
                if (shape.empty() || shape[0] == 0) {
                    continue;
                }

                const size_t resident_experts = std::min<size_t>(shape[0], slot_count);
                const uint64_t bytes_per_expert = (constant_resident_bytes / shape[0]) + (constant_resident_bytes % shape[0] != 0 ? 1 : 0);
                constant_resident_bytes = bytes_per_expert * resident_experts;
            }
            // Per-expert scales and zero points stay resident and also have slot copies
            constant_resident_bytes += get_routed_param_slot_bytes(constant, offload_ratio);

            OPENVINO_ASSERT(constant_resident_bytes <= std::numeric_limits<uint64_t>::max() - resident_bytes, "MoE resident weight size overflows uint64_t");
            resident_bytes += constant_resident_bytes;
        }
    };

    collect_resident_bytes(model);
    return has_moe ? resident_bytes : 0;
}

void validate_model_resident_constant_memory(const ov::Model& model, size_t offload_ratio, uint64_t device_memory_bytes) {
    if (offload_ratio == 0) {
        return;
    }

    // A model without MoE layers is not affected (OFFLOAD_RATIO may be set for every model of a pipeline), but a MoE model
    // where no layer can be offloaded would silently keep every expert resident
    std::string resident_layer;
    for (const auto& node : model.get_ordered_ops()) {
        const auto moe = ov::as_type_ptr<ov::op::internal::MOECompressed>(node);
        if (!moe || !is_otd_capable_moe(moe.get())) {
            continue;
        }
        std::string reason;
        if (get_moe_offload_slot_count(*moe, offload_ratio, &reason) > 0) {
            resident_layer.clear();
            break;
        }
        if (resident_layer.empty()) {
            resident_layer = moe->get_friendly_name() + ": " + reason;
        }
    }
    OPENVINO_ASSERT(resident_layer.empty(), "[GPU] OFFLOAD_RATIO=", offload_ratio, " has no effect: no MoE layer can be offloaded (", resident_layer, ")");

    if (device_memory_bytes == 0) {
        return;
    }

    const uint64_t resident_bytes = get_model_resident_constant_bytes(model, offload_ratio);
    OPENVINO_ASSERT(resident_bytes <= device_memory_bytes,
                    "[GPU] Resident model constants require at least ",
                    resident_bytes,
                    " bytes, exceeding the device global memory size of ",
                    device_memory_bytes,
                    " bytes. Increase OFFLOAD_RATIO or use a GPU with more memory.");
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
    const size_t resident_expert_num = get_routed_weight_slot_count(op, config.get_offload_ratio());
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

#if defined(_WIN32)
namespace {
bool is_luid_empty(const ov::device::LUID& luid) {
    return std::all_of(luid.luid.begin(), luid.luid.end(), [](uint8_t value) { return value == 0; });
}

bool match_luid(const ::LUID& dxgi_luid, const ov::device::LUID& ov_luid) {
    return std::memcmp(&dxgi_luid, ov_luid.luid.data(), sizeof(dxgi_luid)) == 0;
}

uint64_t query_dxgi_available_video_memory_bytes(const ov::device::LUID& luid) {
    if (is_luid_empty(luid))
        return 0;

    using Microsoft::WRL::ComPtr;

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))))
        return 0;

    ComPtr<IDXGIAdapter3> selected_adapter;
    for (UINT adapter_index = 0;; ++adapter_index) {
        ComPtr<IDXGIAdapter> adapter;
        if (factory->EnumAdapters(adapter_index, &adapter) == DXGI_ERROR_NOT_FOUND)
            break;

        DXGI_ADAPTER_DESC desc{};
        if (FAILED(adapter->GetDesc(&desc)))
            continue;

        if (!match_luid(desc.AdapterLuid, luid))
            continue;

        if (SUCCEEDED(adapter.As(&selected_adapter)))
            break;
    }

    if (!selected_adapter)
        return 0;

    uint64_t available_bytes = 0;
    for (UINT node_id = 0;; ++node_id) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (selected_adapter->QueryVideoMemoryInfo(node_id, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info) != S_OK)
            break;

        if (info.Budget > info.CurrentUsage)
            available_bytes += static_cast<uint64_t>(info.Budget - info.CurrentUsage);
    }

    return available_bytes;
}
}  // namespace
#endif

#if defined(__linux__)
uint64_t query_available_ram_bytes() {
    // Read MemAvailable from /proc/meminfo (kernel's official estimate of usable memory)
    std::ifstream meminfo_file("/proc/meminfo");
    if (meminfo_file.is_open()) {
        std::string line;
        while (std::getline(meminfo_file, line)) {
            if (line.rfind("MemAvailable:", 0) == 0) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb_value = 0;
                if (iss >> key >> kb_value) {
                    return kb_value * 1024;
                }
            }
        }
    }
    return 0;
}
#endif

uint64_t estimate_available_tracked_device_memory_bytes(const cldnn::engine& engine, uint64_t upper_bound) {
    uint64_t used_bytes = 0;
    for (const auto& stat : engine.get_memory_statistics()) {
        used_bytes += stat.second;
    }

    if (used_bytes >= upper_bound)
        return 0;

    return upper_bound - used_bytes;
}

namespace {

struct MoEOffloadWeightStats {
    uint64_t total = 0;
    uint64_t routed = 0;
};

// Recursively accumulates weight-constant bytes across the model and any subgraphs.
// w_total counts every Constant once (deduped by node identity); w_moe counts only
// routed-expert Constants (the offloadable subset).
void accumulate_weight_bytes(const ov::Model& model,
                             std::unordered_set<const ov::Node*>& visited,
                             MoEOffloadWeightStats& stats) {
    for (const auto& op : model.get_ops()) {
        if (auto sub = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_model : sub->get_functions()) {
                accumulate_weight_bytes(*sub_model, visited, stats);
            }
        }
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || !visited.insert(constant.get()).second)
            continue;
        stats.total += constant->get_byte_size();
        if (get_moe_constant_role(constant) == MoEConstantRole::OffloadableRoutedExpert)
            stats.routed += constant->get_byte_size();
    }
}

MoEOffloadWeightStats collect_moe_offload_weight_stats(const ov::Model& model) {
    MoEOffloadWeightStats stats;
    std::unordered_set<const ov::Node*> visited;
    accumulate_weight_bytes(model, visited, stats);
    return stats;
}
size_t calculate_auto_offload_ratio(const MoEOffloadWeightStats& stats, uint64_t memory_budget) {
    if (stats.routed == 0 || memory_budget == 0)
        return 0;

    const uint64_t w_fixed = stats.total - stats.routed;

    const double budget_for_moe =
        static_cast<double>(memory_budget) * AUTO_OFFLOAD_RATIO_FIT_SAFETY - static_cast<double>(w_fixed);
    if (budget_for_moe <= 0)
        return 0; // no enough budget for any offload

    if (budget_for_moe >= static_cast<double>(stats.routed)) {
        return 0;  // everything fits, no offload needed
    }

    const double resident_fraction = budget_for_moe / static_cast<double>(stats.routed);
    return static_cast<size_t>(std::lround((1.0 - resident_fraction) * 100.0));
}

}  // namespace

size_t resolve_auto_offload_ratio_for_budget(const ov::Model& model, uint64_t memory_budget) {
    return calculate_auto_offload_ratio(collect_moe_offload_weight_stats(model), memory_budget);
}

size_t resolve_auto_offload_ratio(const ov::Model& model, cldnn::engine& engine) {
    const auto& info = engine.get_device_info();
    const auto stats = collect_moe_offload_weight_stats(model);

    // No offloadable MoE weights -> auto resolves to "no offload".
    if (stats.routed == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] no offloadable MoE routed-expert weights found; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    // Memory budget: device memory for dGPU. For iGPU, prefer OS budget if available and
    // otherwise use the GPU plugin's tracked allocations, as AUTO_BATCH does.
    uint64_t m_budget = info.max_global_mem_size;
    const bool is_igpu = info.dev_type == cldnn::device_type::integrated_gpu;
    std::string budget_source = "device_info";
    if (is_igpu) {
#if defined(_WIN32)
        const uint64_t os_budget = query_dxgi_available_video_memory_bytes(info.luid);
        const char* os_budget_source = "dxgi_budget";
#elif defined(__linux__)
        const uint64_t os_budget = query_available_ram_bytes();
        const char* os_budget_source = "meminfo_budget";
#else
        const uint64_t os_budget = 0;
        const char* os_budget_source = "none";
#endif
        if (os_budget > 0) {
            m_budget = std::min<uint64_t>(m_budget, os_budget);
            budget_source = os_budget_source;
        } else {
            m_budget = estimate_available_tracked_device_memory_bytes(engine, m_budget);
            budget_source = "tracked_mem_stats";
        }
    }
    if (m_budget == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] could not get memory budget; resolved offload_ratio=0" << std::endl;
        return 0; // Failed to offload due to no available memory budget
    }

    const uint64_t w_fixed = stats.total - stats.routed;
    const double budget_for_moe = static_cast<double>(m_budget) * AUTO_OFFLOAD_RATIO_FIT_SAFETY - static_cast<double>(w_fixed);
    if (budget_for_moe <= 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] available memory is insufficient for fixed weights; resolved offload_ratio=0" << std::endl;
        return 0;
    }
    const size_t ratio = calculate_auto_offload_ratio(stats, m_budget);

    GPU_DEBUG_INFO << "[MOE OTD auto] dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
                   << " m_budget=" << m_budget
                   << " budget_source=" << budget_source
                   << " w_total=" << stats.total
                   << " w_moe=" << stats.routed
                   << " w_fixed=" << w_fixed
                   << " budget_for_moe=" << static_cast<long long>(budget_for_moe)
                   << " -> resolved offload_ratio=" << ratio << std::endl;

    if (ratio > AUTO_OFFLOAD_MAX_RATIO_THRESHOLD) {
        GPU_DEBUG_INFO << "[MOE OTD auto] resolved offload_ratio to 0 due to calculated ratio " << ratio << " exceeds " << AUTO_OFFLOAD_MAX_RATIO_THRESHOLD << std::endl;
        return 0; // reject to run with AUTO mode when the calculated offload ratio exceeds the threshold
    }
    return ratio;
}

}  // namespace ov::intel_gpu
