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
#include <functional>
#include <iostream>
#include <limits>
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
#    include <windows.h>
#elif defined(__linux__)
#    include <unistd.h>
#endif
namespace ov::intel_gpu {

// Input indices 3..11 are routed expert weights/scales/zps (WEIGHT_0..ZP_2).
// Input indices 12..21 are shared expert weights (SHARED_GATE_WEIGHT..SHARED_GATE_GATE_WEIGHT).
// See MOE3GemmInputIndex in moe_3gemm_base.hpp for the authoritative enum.
static constexpr size_t ROUTED_INPUT_START = 3;
static constexpr size_t ROUTED_INPUT_END = 11;
static constexpr size_t SHARED_INPUT_START = 12;
static constexpr size_t SHARED_INPUT_END = 21;

bool is_otd_capable_moe(const ov::Node* node) {
    if (const auto* moe = ov::as_type<const ov::op::internal::MOECompressed>(node)) {
        return moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    }
    return false;
}

MoEConstantRole get_moe_constant_role(const std::shared_ptr<ov::op::v0::Constant>& op) {
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

namespace {

// Returns the amount of currently-free physical system RAM in bytes, or 0 if unavailable.
// Used only on integrated GPUs, where device "global memory" is shared with system RAM.
uint64_t get_free_system_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0;
#elif defined(__linux__)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
    return 0;
#else
    return 0;
#endif
}

// Parses a positive size_t from an environment variable, returning default_value if unset/invalid.
size_t env_size_t(const char* name, size_t default_value) {
    const char* v = std::getenv(name);
    if (!v)
        return default_value;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v || parsed == 0)
        return default_value;
    return static_cast<size_t>(parsed);
}

// Parses a double in (0, 1] from an environment variable, returning default_value if unset/invalid.
double env_fraction(const char* name, double default_value) {
    const char* v = std::getenv(name);
    if (!v)
        return default_value;
    char* end = nullptr;
    const double parsed = std::strtod(v, &end);
    if (end == v || !(parsed > 0.0) || parsed > 1.0)
        return default_value;
    return parsed;
}

// Recursively accumulates weight-constant bytes across the model and any subgraphs.
// w_total counts every Constant once (deduped by node identity); w_moe counts only
// routed-expert Constants (the offloadable subset).
void accumulate_weight_bytes(const ov::Model& model,
                             std::unordered_set<const ov::Node*>& visited,
                             uint64_t& w_total,
                             uint64_t& w_moe) {
    for (const auto& op : model.get_ops()) {
        if (auto sub = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_model : sub->get_functions()) {
                accumulate_weight_bytes(*sub_model, visited, w_total, w_moe);
            }
        }
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant)
            continue;
        if (!visited.insert(constant.get()).second)
            continue;  // already counted (shared constant referenced from multiple places)
        const uint64_t bytes = constant->get_byte_size();
        w_total += bytes;
        if (get_moe_constant_role(constant) == MoEConstantRole::RoutedExpert) {
            w_moe += bytes;
        }
    }
}

// Reads a scalar integer from a node input if it is a Constant, returning fallback otherwise.
int64_t read_scalar_input(const ov::Node& op, size_t input_idx, int64_t fallback) {
    if (input_idx >= op.get_input_size())
        return fallback;
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(op.get_input_node_shared_ptr(input_idx));
    if (!c || ov::shape_size(c->get_shape()) < 1)
        return fallback;
    return c->cast_vector<int64_t>().front();
}

// Estimates the runtime memory reserve (KV cache + activations/scratch) from model structure.
// KV cache dominates for LLMs; it is derived from PagedAttention layers using their rt_info
// (k/v head size and kv-head count) and a per-layer effective context length. Layers that use
// sliding-window attention only retain a bounded window of KV, so their context is capped at the
// window size (PagedAttention input #10) rather than the full max_ctx -- this avoids grossly
// over-reserving for hybrid models (e.g. gemma-style 25 sliding + 5 full-attention layers, or
// qwen3.5-moe where only the full-attention layers keep a growing KV cache). Activations/scratch
// are approximated as a fixed fraction of the KV cache. This is model-derived (not a % of device
// memory) so it stays constant across machines of different memory sizes.
// Outputs the number of PagedAttention layers found and the raw KV bytes for diagnostics.
uint64_t estimate_runtime_reserve(const ov::Model& model, size_t max_ctx, size_t& pa_layers, uint64_t& kv_bytes_out) {
    uint64_t kv_bytes = 0;
    pa_layers = 0;
    // PagedAttentionExtension input layout: index 10 is the sliding_window scalar (0 = unlimited).
    constexpr size_t sliding_window_input_idx = 10;
    // Conservative KV element size: assume uncompressed f16 (2 bytes) to over-reserve rather
    // than risk OOM. If KV compression is active the real footprint is smaller, covered by slack.
    constexpr uint64_t kv_elem_bytes = 2;

    std::function<void(const ov::Model&)> walk = [&](const ov::Model& m) {
        for (const auto& op : m.get_ops()) {
            if (auto sub = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
                for (const auto& sub_model : sub->get_functions()) {
                    walk(*sub_model);
                }
            }
            if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
                ++pa_layers;
                const auto& rt = op->get_rt_info();
                auto read = [&](const char* key) -> int64_t {
                    auto it = rt.find(key);
                    return it == rt.end() ? 0 : it->second.as<int64_t>();
                };
                const int64_t k_head_size = read("k_head_size");
                const int64_t v_head_size = read("v_head_size");
                const int64_t num_k_heads = read("num_k_heads");
                if (k_head_size > 0 && v_head_size > 0 && num_k_heads > 0) {
                    // Sliding-window layers only retain a bounded KV window; cap the effective
                    // context at the window size when present (>0). 0 means unlimited (full ctx).
                    uint64_t eff_ctx = max_ctx;
                    const int64_t window = read_scalar_input(*op, sliding_window_input_idx, 0);
                    if (window > 0) {
                        eff_ctx = std::min<uint64_t>(eff_ctx, static_cast<uint64_t>(window));
                    }
                    // Per layer: (K + V) * num_kv_heads * eff_ctx * bytes.
                    kv_bytes += static_cast<uint64_t>(k_head_size + v_head_size) * static_cast<uint64_t>(num_k_heads) *
                                eff_ctx * kv_elem_bytes;
                }
            }
        }
    };
    walk(model);

    kv_bytes_out = kv_bytes;
    // Add ~25% on top of KV for activations and scratch buffers.
    return kv_bytes + kv_bytes / 4;
}

}  // namespace

size_t resolve_auto_offload_ratio(const ov::Model& model, const cldnn::device_info& info) {
    uint64_t w_total = 0;
    uint64_t w_moe = 0;
    std::unordered_set<const ov::Node*> visited;
    accumulate_weight_bytes(model, visited, w_total, w_moe);

    // No offloadable MoE weights -> auto resolves to "no offload".
    if (w_moe == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] no offloadable MoE routed-expert weights found; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    // Memory budget: device memory for dGPU; for iGPU cap by free system RAM since the
    // device "global memory" is shared with (and overstated relative to) actual free RAM.
    uint64_t m_budget = info.max_global_mem_size;
    const bool is_igpu = info.dev_type == cldnn::device_type::integrated_gpu;
    if (is_igpu) {
        const uint64_t free_ram = get_free_system_ram_bytes();
        if (free_ram > 0) {
            m_budget = std::min<uint64_t>(m_budget, free_ram);
        }
    }
    if (m_budget == 0) {
        GPU_DEBUG_INFO << "[MOE OTD auto] could not determine memory budget; resolved offload_ratio=0" << std::endl;
        return 0;
    }

    const size_t max_ctx = 8192;
    const double safety = 0.85;
    const uint64_t w_fixed = w_total - w_moe;
    size_t pa_layers = 0;
    uint64_t kv_bytes = 0;
    const uint64_t reserve = estimate_runtime_reserve(model, max_ctx, pa_layers, kv_bytes);
    const uint64_t free_ram_now = get_free_system_ram_bytes();

    const double budget_for_moe =
        static_cast<double>(m_budget) * safety - static_cast<double>(w_fixed) - static_cast<double>(reserve);

    size_t ratio;
    if (budget_for_moe >= static_cast<double>(w_moe)) {
        ratio = 0;  // everything fits, no offload
    } else if (budget_for_moe <= 0.0) {
        ratio = 99;  // extreme pressure: keep a single LRU slot resident (100 == all-on-disk is invalid)
    } else {
        const double resident_fraction = budget_for_moe / static_cast<double>(w_moe);
        const long r = std::lround((1.0 - resident_fraction) * 100.0);
        ratio = static_cast<size_t>(std::clamp<long>(r, 0, 99));
    }

    constexpr double to_mib = 1.0 / (1024.0 * 1024.0);
    // Emit unconditionally (compile-time, once per model) so the decision is visible even in
    // release builds where GPU_DEBUG_INFO is compiled out.
    std::cout << "[MOE OTD auto] dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
              << " dev_mem=" << static_cast<uint64_t>(info.max_global_mem_size * to_mib) << "MiB"
              << " free_ram_now=" << static_cast<uint64_t>(free_ram_now * to_mib) << "MiB"
              << " M_budget=" << static_cast<uint64_t>(m_budget * to_mib) << "MiB"
              << " W_total=" << static_cast<uint64_t>(w_total * to_mib) << "MiB"
              << " W_moe=" << static_cast<uint64_t>(w_moe * to_mib) << "MiB"
              << " W_fixed=" << static_cast<uint64_t>(w_fixed * to_mib) << "MiB"
              << " Reserve=" << static_cast<uint64_t>(reserve * to_mib) << "MiB"
              << " (pa_layers=" << pa_layers << " kv=" << static_cast<uint64_t>(kv_bytes * to_mib) << "MiB)"
              << " budget_for_moe=" << static_cast<long long>(budget_for_moe * to_mib) << "MiB"
              << " SAFETY=" << safety << " max_ctx=" << max_ctx
              << " -> offload_ratio=" << ratio << std::endl;

    GPU_DEBUG_INFO << "[MOE OTD auto] M_budget=" << m_budget << " (dev_type=" << (info.dev_type == cldnn::device_type::integrated_gpu ? "iGPU" : "dGPU")
                   << "), W_total=" << w_total << ", W_moe=" << w_moe << ", W_fixed=" << w_fixed << ", Reserve=" << reserve
                   << ", SAFETY=" << safety << ", max_ctx=" << max_ctx << " -> resolved offload_ratio=" << ratio << std::endl;
    return ratio;
}

}  // namespace ov::intel_gpu
