// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common_utils/gpu_execution_plan.hpp"
#include "common_utils/gpu_kernel_lifecycle.hpp"
#include "eltwise/fusion_analysis.hpp"
#include "eltwise/kernel_catalog.hpp"
#include "eltwise/kernel_kind.hpp"
#include "eltwise/kernel_selection.hpp"
#include "eltwise/local_size_tuner.hpp"
#include "eltwise/metadata_builder.hpp"
#include "eltwise/operation_semantics.hpp"
#include "eltwise_shader_abi.hpp"
#include "graph_optimizer/vulkan_graph_optimizer.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "registry/implementation_map.hpp"
#include "shader_scalar_type.hpp"
#include "vulkan/vulkan_engine.hpp"
#include "vulkan/vulkan_memory.hpp"
#include "vulkan/vulkan_stream.hpp"
#include "vulkan_shader_abi.hpp"

namespace cldnn {
namespace vulkan {
namespace {

[[maybe_unused]] const bool graph_optimizer_registered = [] {
    register_graph_optimizer();
    return true;
}();

namespace shader_abi = eltwise_shader_abi;
using namespace eltwise_detail;

constexpr uint32_t broadcast_vector_width = 4;
constexpr uint32_t broadcast_vector_min_elements = 256;
constexpr uint32_t maximum_scalar_batch_width = 8;
constexpr uint32_t balanced_scalar_batch_width = 4;
constexpr uint32_t division_scalar_batch_width = 2;
constexpr uint32_t scalar_batch_width = 1;
constexpr uint32_t scalar_elements_per_subgroup_budget = portable_local_work_group_size_limit;

uint32_t operation_batch_width_limit(eltwise_mode mode, data_types type, kernel_kind kind, const device_info& info) {
    switch (mode) {
    case eltwise_mode::pow:
    case eltwise_mode::atan2:
        return scalar_batch_width;
    case eltwise_mode::div:
    case eltwise_mode::mod:
    case eltwise_mode::floor_mod:
        if (type == data_types::i64) {
            return scalar_batch_width;
        }
        if (is_broadcast_vector_kernel(kind) && info.max_work_group_size != 0) {
            const auto capability_width = std::max<uint64_t>(info.max_work_group_size / portable_local_work_group_size_limit, scalar_batch_width);
            return static_cast<uint32_t>(std::min<uint64_t>(capability_width, balanced_scalar_batch_width));
        }
        return division_scalar_batch_width;
    case eltwise_mode::squared_diff:
        return balanced_scalar_batch_width;
    default:
        return maximum_scalar_batch_width;
    }
}

bool has_aligned_offset(const layout& tensor_layout, uint32_t width) {
    return tensor_layout.get_linear_offset() % width == 0;
}

uint32_t select_generic_elements_per_invocation(kernel_kind kind,
                                                const layout& input0_layout,
                                                const layout& input1_layout,
                                                const layout& output_layout,
                                                const layout* fused_input_layout,
                                                eltwise_mode mode,
                                                const std::vector<eltwise_mode>& fused_modes,
                                                const device_info& info) {
    const auto packed_width = get_packed_dense_width(kind);
    if (packed_width != packed_dense_width::none) {
        return value(packed_width);
    }
    const auto vector_width = get_dense_vector_width(kind);
    if (vector_width != dense_vector_width::scalar) {
        return value(vector_width);
    }
    const auto output_width = output_elements_per_invocation(output_layout);
    if (output_width != 1 || kind == kernel_kind::unary || kind == kernel_kind::broadcast_scalar || kind == kernel_kind::broadcast_fast_scalar) {
        return output_width;
    }
    OPENVINO_ASSERT(is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind) || is_broadcast_vector_kernel(kind) ||
                        kind == kernel_kind::scalar_constant,
                    "[GPU][Vulkan] Eltwise scalar batch selection received an unsupported kernel kind");

    uint32_t width_limit = std::min({operation_batch_width_limit(mode, input0_layout.data_type, kind, info),
                                     operation_batch_width_limit(mode, input1_layout.data_type, kind, info),
                                     operation_batch_width_limit(mode, output_layout.data_type, kind, info)});
    for (const auto fused_mode : fused_modes) {
        width_limit = std::min(width_limit, operation_batch_width_limit(fused_mode, output_layout.data_type, kind, info));
    }
    if (is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind) || kind == kernel_kind::scalar_constant) {
        width_limit = std::min(width_limit, balanced_scalar_batch_width);
    }
    auto maximum_scalar_size = std::max({data_type_traits::size_of(input0_layout.data_type),
                                         data_type_traits::size_of(input1_layout.data_type),
                                         data_type_traits::size_of(output_layout.data_type)});
    if (fused_input_layout != nullptr) {
        maximum_scalar_size = std::max(maximum_scalar_size, data_type_traits::size_of(fused_input_layout->data_type));
    }
    if (maximum_scalar_size >= sizeof(uint64_t) && width_limit >= division_scalar_batch_width) {
        return division_scalar_batch_width;
    }
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0 || info.max_work_group_size == 0) {
        return scalar_batch_width;
    }

    const auto subgroup_size = static_cast<uint64_t>(info.supported_simd_sizes.front());
    const auto subgroup_width_limit = std::max<uint64_t>(scalar_elements_per_subgroup_budget / subgroup_size, balanced_scalar_batch_width);
    width_limit = static_cast<uint32_t>(std::min<uint64_t>(width_limit, subgroup_width_limit));
    const auto max_work_group_size = std::max<uint64_t>(info.max_work_group_size, 1);
    const auto subgroups_per_full_work_group = (max_work_group_size + subgroup_size - 1) / subgroup_size;
    const auto useful_subgroup_coverage = std::min<uint64_t>(subgroups_per_full_work_group, width_limit);
    const auto required_parallel_invocations = max_work_group_size * useful_subgroup_coverage;
    const auto has_aligned_layouts = [&](uint32_t candidate) {
        if (!has_aligned_offset(output_layout, candidate)) {
            return false;
        }
        const bool requires_aligned_inputs = is_plain_dense_kernel(kind) || is_fused_kernel(kind) || is_fused_post_op_kernel(kind);
        return !requires_aligned_inputs ||
               (has_aligned_offset(input0_layout, candidate) && has_aligned_offset(input1_layout, candidate) &&
                (fused_input_layout == nullptr || is_fused_broadcast_kernel(kind) || has_aligned_offset(*fused_input_layout, candidate)));
    };
    constexpr std::array candidates{maximum_scalar_batch_width, balanced_scalar_batch_width, division_scalar_batch_width, scalar_batch_width};
    for (const auto candidate : candidates) {
        if (candidate > width_limit || !has_aligned_layouts(candidate)) {
            continue;
        }
        const auto invocation_count = (output_layout.count() + candidate - 1) / candidate;
        if (invocation_count >= required_parallel_invocations) {
            return candidate;
        }
    }
    return scalar_batch_width;
}

bool can_use_broadcast_vector_kernel(const layout& output_layout) {
    if (data_type_traits::size_of(output_layout.data_type) < sizeof(uint32_t) || output_layout.count() < broadcast_vector_min_elements) {
        return false;
    }
    return output_layout.count() % broadcast_vector_width == 0;
}

bool should_use_fast_broadcast_kernel(const layout& input0_layout,
                                      const layout& input1_layout,
                                      const layout& output_layout,
                                      const device_info& info,
                                      uint32_t elements_per_invocation) {
    if (info.supported_simd_sizes.empty() || info.supported_simd_sizes.front() == 0 || info.max_work_group_size == 0) {
        return false;
    }
    const auto subgroup_size = info.supported_simd_sizes.front();
    const auto subgroup_slots = std::max<uint64_t>(info.max_work_group_size / subgroup_size, 1);
    const auto rank = EltwiseMetadata::collapsed_broadcast_rank(input0_layout, input1_layout, output_layout);
    const auto coordinate_schedule_pressure = static_cast<uint64_t>(rank) * subgroup_slots;
    const auto invocation_count = (output_layout.count() + elements_per_invocation - 1) / elements_per_invocation;
    return coordinate_schedule_pressure <= portable_local_work_group_size_limit && invocation_count >= subgroup_size;
}

bool output_is_disjoint_from_reads(const memory::cptr& output, const std::vector<memory::cptr>& reads) {
    const auto* output_buffer = output == nullptr ? nullptr : dynamic_cast<const vulkan_buffer*>(output.get());
    if (output_buffer == nullptr) {
        return false;
    }

    const auto output_begin = output_buffer->get_offset();
    const auto output_end = output_begin + output_buffer->size();
    for (const auto& read : reads) {
        const auto* read_buffer = read == nullptr ? nullptr : dynamic_cast<const vulkan_buffer*>(read.get());
        if (read_buffer == nullptr) {
            return false;
        }
        if (output_buffer->get_allocation() != read_buffer->get_allocation()) {
            continue;
        }
        const auto read_begin = read_buffer->get_offset();
        const auto read_end = read_begin + read_buffer->size();
        if (std::max(output_begin, read_begin) < std::min(output_end, read_end)) {
            return false;
        }
    }
    return true;
}

struct eltwise_impl : typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::eltwise_impl)

    eltwise_impl() : eltwise_impl(kernel_kind::broadcast_scalar, std::nullopt, 0, 0) {}

    eltwise_impl(kernel_kind kind, std::optional<scalar_constant> scalar, uint32_t elements_per_invocation, uint32_t fused_chain_length)
        : parent("vulkan_eltwise"),
          _kernel_sources(make_kernel_sources(kind, fused_chain_length)),
          _kernel_kind(kind),
          _scalar_constant(std::move(scalar)),
          _elements_per_invocation(elements_per_invocation),
          _fused_chain_length(fused_chain_length),
          _local_size_tuner(std::make_shared<LocalSizeTuner>()),
          _execution_plan(1) {}

    std::unique_ptr<primitive_impl> clone() const override {
        auto result = std::make_unique<eltwise_impl>(*this);
        result->_metadata_initialized = false;
        result->_local_size_tuner = std::make_shared<LocalSizeTuner>();
        return result;
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    void save(BinaryOutputBuffer& buffer) const override {
        parent::save(buffer);
        buffer << make_data(&_kernel_kind, sizeof(_kernel_kind));
        buffer << make_data(&_elements_per_invocation, sizeof(_elements_per_invocation));
        buffer << make_data(&_fused_chain_length, sizeof(_fused_chain_length));
        const bool has_scalar_constant = _scalar_constant.has_value();
        buffer << make_data(&has_scalar_constant, sizeof(has_scalar_constant));
        if (has_scalar_constant) {
            buffer << make_data(&_scalar_constant->input_index, sizeof(_scalar_constant->input_index));
            buffer << make_data(_scalar_constant->bits.data(), sizeof(_scalar_constant->bits));
        }
    }

    void load(BinaryInputBuffer& buffer) override {
        parent::load(buffer);
        buffer >> make_data(&_kernel_kind, sizeof(_kernel_kind));
        buffer >> make_data(&_elements_per_invocation, sizeof(_elements_per_invocation));
        buffer >> make_data(&_fused_chain_length, sizeof(_fused_chain_length));
        bool has_scalar_constant = false;
        buffer >> make_data(&has_scalar_constant, sizeof(has_scalar_constant));
        if (has_scalar_constant) {
            _scalar_constant.emplace();
            buffer >> make_data(&_scalar_constant->input_index, sizeof(_scalar_constant->input_index));
            buffer >> make_data(_scalar_constant->bits.data(), sizeof(_scalar_constant->bits));
        } else {
            _scalar_constant.reset();
        }
        _kernel_sources = make_kernel_sources(_kernel_kind, _fused_chain_length);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        if (uses_dense_push_constants(_kernel_kind)) {
            return {};
        }
        return {BufferDescriptor(EltwiseMetadata::buffer_word_count(_kernel_kind), ov::element::u32, true, false)};
    }

    void init_kernels(const kernels_cache& cache, const kernel_impl_params& params) override {
        this->can_share_kernels = _kernels.initialize(cache, params);
        OPENVINO_ASSERT(_kernels.size() == _kernel_sources.size(), "[GPU][Vulkan] Eltwise compiled an unexpected number of SPIR-V kernels");
    }

    void init_by_cached_kernels(const kernels_cache& cache, std::vector<std::string>& cached_kernel_ids) override {
        this->can_share_kernels = _kernels.restore(cache, cached_kernel_ids);
        OPENVINO_ASSERT(_kernels.size() == 1 || (supports_restricted_output(_kernel_kind) && _kernels.size() == 2),
                        "[GPU][Vulkan] Cached Eltwise expects an alias-safe kernel and an optional restricted kernel");
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) override {
        return _kernels.get_cached_kernel_ids(cache);
    }

    std::vector<std::shared_ptr<kernel_string>> get_kernels_source() override {
        return _kernel_sources;
    }

    void reset_kernels_source() override {
        _kernel_sources.clear();
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels.copy_kernels();
    }

    std::vector<size_t> get_in_place_input_indices() const override {
        return supports_restricted_output(_kernel_kind) ? std::vector<size_t>{0, 1} : std::vector<size_t>{};
    }

    void set_kernels(kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "[GPU][Vulkan] Eltwise expects one compiled kernel set");
        OPENVINO_ASSERT(kernels.begin()->second.size() == _kernel_sources.size(), "[GPU][Vulkan] Eltwise compiled an unexpected number of kernels");
        _kernels.adopt_compiled(std::move(kernels));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OPENVINO_ASSERT((_kernels.size() == 1 || _kernels.size() == 2) && _kernels.front() != nullptr, "[GPU][Vulkan] Eltwise kernel was not initialized");
        const auto desc = instance.get_typed_desc<eltwise>();
        const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
        const auto fused = get_supported_fused_eltwise_chain(instance.get_node());
        const auto post_op = get_supported_fused_post_op(instance.get_node());
        const bool use_fused_kernel = is_fused_kernel(_kernel_kind);
        const bool use_fused_post_op_kernel = is_fused_post_op_kernel(_kernel_kind);
        const bool use_fused_dense_kernel = is_fused_dense_kernel(_kernel_kind);
        const bool use_fused_broadcast_kernel = is_fused_broadcast_kernel(_kernel_kind);
        const bool use_single_fused_kernel = is_single_fused_kernel(_kernel_kind);
        const bool use_fused_chain_kernel = is_fused_chain_kernel(_kernel_kind);
        const bool use_dense_push_constants = uses_dense_push_constants(_kernel_kind);
        OPENVINO_ASSERT(use_fused_kernel == fused.has_value(), "[GPU][Vulkan] Fused Eltwise kernel and graph metadata are inconsistent");
        OPENVINO_ASSERT(use_fused_post_op_kernel == post_op.has_value(), "[GPU][Vulkan] Fused Eltwise post-op kernel and graph metadata are inconsistent");
        OPENVINO_ASSERT(!use_single_fused_kernel || fused->size() == 1, "[GPU][Vulkan] Single-stage fused Eltwise kernel received a longer chain");
        OPENVINO_ASSERT(!use_fused_broadcast_kernel || fused->front().broadcast_input,
                        "[GPU][Vulkan] Fused broadcast Eltwise kernel received a dense external input");
        OPENVINO_ASSERT(!use_fused_chain_kernel || (fused->size() >= EltwiseMetadata::minimum_multi_stage_fused_chain_length &&
                                                    fused->size() <= EltwiseMetadata::maximum_fused_chain_length && fused->size() == _fused_chain_length),
                        "[GPU][Vulkan] Fused Eltwise chain kernel received an unsupported chain length");
        OPENVINO_ASSERT(use_fused_chain_kernel || _fused_chain_length == 0, "[GPU][Vulkan] Non-chain Eltwise kernel received fixed chain metadata");
        OPENVINO_ASSERT(instance.inputs_memory_count() == base_input_count && instance.get_fused_mem_count() == (fused.has_value() ? fused->size() : 0U),
                        "[GPU][Vulkan] Eltwise received an unexpected regular or fused input count");
        OPENVINO_ASSERT(instance.get_intermediates_memories().size() == (use_dense_push_constants ? 0U : 1U),
                        "[GPU][Vulkan] Eltwise metadata resources do not match the selected kernel ABI");

        auto& stream = instance.get_network().get_stream();
        const auto metadata = EltwiseMetadata::build(instance, _scalar_constant, fused, post_op);
        memory::ptr metadata_memory;
        if (!use_dense_push_constants) {
            metadata_memory = instance.get_intermediates_memories().front();
            if (!_metadata_initialized || metadata != _cached_metadata) {
                metadata_memory->copy_from(stream, metadata.data(), true);
                _cached_metadata = metadata;
                _metadata_initialized = true;
            }
        }

        kernel_arguments_desc descriptor;
        descriptor.layerID = instance.id();
        const auto element_count = metadata[shader_abi::index(shader_abi::metadata_field::element_count)];
        const auto& input0_layout = instance.get_input_layout(0);
        const auto& input1_layout = base_input_count == 1 ? input0_layout : instance.get_input_layout(1);
        const auto& output_layout = instance.get_output_layout(0);
        const bool use_dense_kernel = is_plain_dense_kernel(_kernel_kind) || use_fused_kernel || use_fused_post_op_kernel;
        const auto* fused_input_layout = use_fused_kernel ? &instance.get_input_layout(fused->front().external_dependency_index) : nullptr;
        const auto selected_dense_vector_width = get_dense_vector_width(_kernel_kind);
        const bool use_broadcast_vector_kernel = is_broadcast_vector_kernel(_kernel_kind);
        const bool use_fast_broadcast_kernel = is_fast_broadcast_kernel(_kernel_kind);
        const bool use_unary_kernel = _kernel_kind == kernel_kind::unary;
        const bool use_scalar_constant_kernel = _kernel_kind == kernel_kind::scalar_constant;
        OPENVINO_ASSERT(use_scalar_constant_kernel == _scalar_constant.has_value(),
                        "[GPU][Vulkan] Scalar Eltwise kernel and constant metadata are inconsistent");
        OPENVINO_ASSERT(!is_plain_dense_kernel(_kernel_kind) ||
                            (is_packed_dense_kernel(_kernel_kind) ? can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, nullptr)
                                                                  : can_use_dense_kernel(input0_layout, input1_layout, output_layout)),
                        "[GPU][Vulkan] Dense Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_fused_dense_kernel ||
                            (is_packed_dense_kernel(_kernel_kind) ? can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, fused_input_layout)
                                                                  : std::all_of(fused->begin(),
                                                                                fused->end(),
                                                                                [&](const fused_eltwise_info& fused_stage) {
                                                                                    return can_use_fused_dense_kernel(
                                                                                        input0_layout,
                                                                                        input1_layout,
                                                                                        instance.get_input_layout(fused_stage.external_dependency_index),
                                                                                        output_layout);
                                                                                })),
                        "[GPU][Vulkan] Fused dense Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_fused_broadcast_kernel || (fused->size() == 1 && fused->front().broadcast_input && has_dense_storage(*fused_input_layout) &&
                                                        is_numpy_broadcast_compatible(*fused_input_layout, output_layout)),
                        "[GPU][Vulkan] Fused broadcast Eltwise runtime layouts no longer satisfy the compiled "
                        "kernel contract");
        OPENVINO_ASSERT(!use_fused_post_op_kernel || can_use_fused_post_op_kernel(input0_layout, input1_layout, *post_op, output_layout),
                        "[GPU][Vulkan] Fused Eltwise post-op runtime layouts no longer satisfy the compiled "
                        "kernel contract");
        OPENVINO_ASSERT(selected_dense_vector_width == dense_vector_width::scalar ||
                            can_use_f32_dense_vector_width(input0_layout, input1_layout, output_layout, fused_input_layout, selected_dense_vector_width),
                        "[GPU][Vulkan] F32 vector Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        OPENVINO_ASSERT(!use_broadcast_vector_kernel || can_use_broadcast_vector_kernel(output_layout),
                        "[GPU][Vulkan] Vector broadcast Eltwise runtime layout no longer satisfies the compiled kernel contract");
        auto elements_per_invocation = _elements_per_invocation;
        if (elements_per_invocation == 0) {
            std::vector<eltwise_mode> fused_modes;
            if (fused.has_value()) {
                fused_modes.reserve(fused->size());
                for (const auto& fused_stage : *fused) {
                    fused_modes.push_back(fused_stage.descriptor->typed_desc<eltwise>()->mode);
                }
            }
            elements_per_invocation = select_generic_elements_per_invocation(_kernel_kind,
                                                                             input0_layout,
                                                                             input1_layout,
                                                                             output_layout,
                                                                             fused_input_layout,
                                                                             desc->mode,
                                                                             fused_modes,
                                                                             instance.get_network().get_engine().get_device_info());
        }
        OPENVINO_ASSERT(!use_fast_broadcast_kernel || should_use_fast_broadcast_kernel(input0_layout,
                                                                                       input1_layout,
                                                                                       output_layout,
                                                                                       instance.get_network().get_engine().get_device_info(),
                                                                                       elements_per_invocation),
                        "[GPU][Vulkan] Fast broadcast Eltwise runtime layouts no longer satisfy the compiled kernel contract");
        const auto invocation_count = (element_count + elements_per_invocation - 1) / elements_per_invocation;
        const auto& device_info = instance.get_network().get_engine().get_device_info();
        std::optional<LocalSizeTuner::Selection> local_size_selection;
        uint32_t local_size = 1;
        if (_elements_per_invocation != 0) {
            local_size = _local_size_tuner->cached_local_size();
            if (local_size == 0) {
                local_size_selection.emplace(_local_size_tuner->select_uncached(invocation_count, device_info, is_no_tail_kernel(_kernel_kind)));
                local_size = local_size_selection->local_size();
            }
        } else {
            local_size = select_portable_local_work_group_size(invocation_count, device_info.max_work_group_size);
        }
        descriptor.workGroups.global = {invocation_count, 1, 1};
        descriptor.workGroups.local = {local_size, 1, 1};
        vulkan_specialization_constants specialization_constants = {
            {cldnn::vulkan::shader_abi::index(cldnn::vulkan::shader_abi::specialization_id::local_size_x), local_size},
            {shader_abi::index(shader_abi::specialization_id::mode), metadata[shader_abi::index(shader_abi::metadata_field::mode)]},
            {shader_abi::index(shader_abi::specialization_id::input0_type), metadata[shader_abi::index(shader_abi::metadata_field::input0_type)]},
            {shader_abi::index(shader_abi::specialization_id::input1_type), metadata[shader_abi::index(shader_abi::metadata_field::input1_type)]},
            {shader_abi::index(shader_abi::specialization_id::output_type), metadata[shader_abi::index(shader_abi::metadata_field::output_type)]},
            {shader_abi::index(shader_abi::specialization_id::storage_flags), metadata[shader_abi::index(shader_abi::metadata_field::flags)]},
        };
        if (use_dense_kernel || use_broadcast_vector_kernel || use_scalar_constant_kernel) {
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::elements_per_invocation), elements_per_invocation});
        }
        if (use_fast_broadcast_kernel) {
            specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_rank), metadata[shader_abi::index(shader_abi::metadata_field::rank)]});
            specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_input0_axes), metadata.active_broadcast_axes(shader_abi::tensor_index::input0)});
            specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_input1_axes), metadata.active_broadcast_axes(shader_abi::tensor_index::input1)});
            specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::broadcast_output_axes), metadata.active_broadcast_axes(shader_abi::tensor_index::output)});
        }
        if (use_single_fused_kernel) {
            specialization_constants.push_back(
                {shader_abi::index(shader_abi::specialization_id::fused_mode), metadata.fused_stage_value(0, shader_abi::fused_metadata_field::mode)});
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_input_type),
                                                metadata.fused_stage_value(0, shader_abi::fused_metadata_field::input_type)});
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_input_position),
                                                metadata.fused_stage_value(0, shader_abi::fused_metadata_field::input_position)});
        } else if (use_fused_chain_kernel) {
            for (size_t stage = 0; stage < fused->size(); ++stage) {
                const auto stage_index = checked_u32(stage, "fused specialization stage");
                specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_chain_mode_base) + stage_index,
                                                    metadata.fused_stage_value(stage, shader_abi::fused_metadata_field::mode)});
                specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_chain_input_type_base) + stage_index,
                                                    metadata.fused_stage_value(stage, shader_abi::fused_metadata_field::input_type)});
                specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::fused_chain_input_position_base) + stage_index,
                                                    metadata.fused_stage_value(stage, shader_abi::fused_metadata_field::input_position)});
            }
        }
        if (use_fused_post_op_kernel) {
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::base_output_type),
                                                shader_abi::value(to_shader_scalar_type(post_op->descriptor->input_layout.data_type))});
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::post_op_kind), shader_abi::value(post_op->kind)});
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::post_activation), shader_abi::value(post_op->activation)});
            specialization_constants.push_back({shader_abi::index(shader_abi::specialization_id::post_quantize_flags), post_op->quantize_flags});
        }
        if (use_dense_push_constants) {
            descriptor.scalars = metadata.make_dense_push_constants(is_single_fused_dense_kernel(_kernel_kind));
        }
        descriptor.arguments = {{argument_desc::Types::INPUT, 0}};
        if (use_scalar_constant_kernel) {
            descriptor.arguments.front().index = _scalar_constant->input_index == shader_abi::tensor_index::input0
                                                     ? shader_abi::index(shader_abi::tensor_index::input1)
                                                     : shader_abi::index(shader_abi::tensor_index::input0);
        } else if (!use_unary_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::INPUT, 1});
        }
        if (use_single_fused_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
        } else if (use_fused_chain_kernel) {
            for (size_t stage = 0; stage < EltwiseMetadata::maximum_fused_chain_length; ++stage) {
                descriptor.arguments.push_back({argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE, static_cast<uint32_t>(std::min(stage, fused->size() - 1))});
            }
        }
        descriptor.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        if (!use_dense_push_constants) {
            descriptor.arguments.push_back({argument_desc::Types::INTERNAL_BUFFER, 0});
        }
        if (!use_dense_kernel) {
            descriptor.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        }

        kernel_arguments_data arguments;
        for (size_t input_index = 0; input_index < base_input_count; ++input_index) {
            arguments.inputs.push_back(instance.input_memory_ptr(input_index));
        }
        if (use_fused_kernel) {
            for (size_t stage = 0; stage < fused->size(); ++stage) {
                arguments.fused_op_inputs.push_back(instance.fused_memory(stage));
            }
        }
        arguments.outputs = {instance.output_memory_ptr(0)};
        if (!use_dense_push_constants) {
            arguments.intermediates = {metadata_memory};
        }
        std::vector<memory::cptr> read_memories;
        if (use_scalar_constant_kernel) {
            const auto tensor_input_index = _scalar_constant->input_index == shader_abi::tensor_index::input0
                                                ? shader_abi::index(shader_abi::tensor_index::input1)
                                                : shader_abi::index(shader_abi::tensor_index::input0);
            read_memories.push_back(arguments.inputs.at(tensor_input_index));
        } else {
            read_memories.insert(read_memories.end(), arguments.inputs.begin(), arguments.inputs.end());
        }
        read_memories.insert(read_memories.end(), arguments.fused_op_inputs.begin(), arguments.fused_op_inputs.end());
        read_memories.insert(read_memories.end(), arguments.intermediates.begin(), arguments.intermediates.end());
        const auto use_restricted_kernel = _kernels.size() == 2 && output_is_disjoint_from_reads(arguments.outputs.front(), read_memories);
        const size_t selected_kernel_index = use_restricted_kernel ? 1 : 0;
        _execution_plan[0].kernel_index = selected_kernel_index;
        auto& selected_kernel = *_kernels.at(selected_kernel_index);
        if (local_size_selection.has_value() && local_size_selection->requires_prewarm()) {
            for (const auto candidate : local_size_selection->candidates()) {
                descriptor.workGroups.local[0] = candidate;
                stream.set_arguments(selected_kernel, descriptor, arguments);
            }
            descriptor.workGroups.local[0] = local_size_selection->complete_prewarm();
        }
        if (local_size_selection.has_value() && local_size_selection->requires_measurement()) {
            stream.finish();
            const auto start = std::chrono::steady_clock::now();
            auto& vulkan_dispatch_stream = dynamic_cast<vulkan_stream&>(stream);
            auto measured_event = vulkan_dispatch_stream.enqueue_kernel(selected_kernel, descriptor, arguments, specialization_constants, events, true);
            OPENVINO_ASSERT(measured_event != nullptr, "[GPU][Vulkan] Eltwise local-size measurement did not produce a completion event");
            measured_event->wait();
            const auto elapsed = std::chrono::steady_clock::now() - start;
            local_size_selection->complete_measurement(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
            return measured_event;
        }
        auto& vulkan_dispatch_stream = dynamic_cast<vulkan_stream&>(stream);
        return _execution_plan.execute_with(
            stream,
            _kernels,
            events,
            instance.needs_completion_event(),
            [&](size_t) {
                return gpu_dispatch_binding{&descriptor, std::move(arguments)};
            },
            [&](size_t,
                kernel& selected_kernel,
                const kernel_arguments_desc& kernel_descriptor,
                const kernel_arguments_data& kernel_arguments,
                const std::vector<event::ptr>& dependencies,
                bool request_completion) {
                return vulkan_dispatch_stream
                    .enqueue_kernel(selected_kernel, kernel_descriptor, kernel_arguments, specialization_constants, dependencies, request_completion);
            });
    }

private:
    std::vector<std::shared_ptr<kernel_string>> _kernel_sources;
    kernel_kind _kernel_kind = kernel_kind::broadcast_scalar;
    std::optional<scalar_constant> _scalar_constant;
    uint32_t _elements_per_invocation = 0;
    uint32_t _fused_chain_length = 0;
    std::shared_ptr<LocalSizeTuner> _local_size_tuner;
    gpu_kernel_lifecycle _kernels;
    gpu_execution_plan _execution_plan;
    EltwiseMetadata _cached_metadata;
    bool _metadata_initialized = false;
};

}  // namespace

bool EltwiseImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }
    const auto& desc = node.as<eltwise>().get_primitive();
    const auto expected_inputs = is_unary_mode(desc->mode) ? 1U : 2U;
    const auto fused = get_supported_fused_eltwise_chain(node);
    const auto post_op = get_supported_fused_post_op(node);
    if (node.has_fused_primitives() && !fused.has_value() && !post_op.has_value()) {
        return false;
    }
    const bool coefficients_supported = desc->coefficients.empty() || (desc->mode == eltwise_mode::sum && desc->coefficients.size() == expected_inputs) ||
                                        (desc->mode == eltwise_mode::is_inf && desc->coefficients.size() == 2);
    if (!is_supported_mode(desc->mode) || node.get_dependencies().size() != expected_inputs + (fused.has_value() ? fused->size() : 0U) ||
        !coefficients_supported || !desc->stride.empty() ||
        (desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NUMPY && desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NONE)) {
        return false;
    }

    for (size_t index = 0; index < expected_inputs; ++index) {
        const auto& input_layout = node.get_input_layout(index);
        if (!is_supported_shader_scalar_type(input_layout.data_type) || !is_supported_format(input_layout.format.value) ||
            input_layout.get_rank() > EltwiseMetadata::maximum_rank) {
            return false;
        }
        if (is_bitwise_mode(desc->mode) && !is_integer_shader_scalar_type(input_layout.data_type)) {
            return false;
        }
        if ((desc->mode == eltwise_mode::atan2 || is_unary_mode(desc->mode)) && !data_type_traits::is_floating_point(input_layout.data_type)) {
            return false;
        }
    }
    const auto& output_layout = node.get_output_layout(0);
    return is_supported_shader_scalar_type(output_layout.data_type) && is_supported_format(output_layout.format.value) &&
           output_layout.get_rank() <= EltwiseMetadata::maximum_rank;
}

std::unique_ptr<primitive_impl> EltwiseImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    const auto* engine = dynamic_cast<const vulkan_engine*>(&node.get_program().get_engine());
    OPENVINO_ASSERT(engine != nullptr, "[GPU][Vulkan] Eltwise implementation requires a Vulkan engine");
    const auto max_push_constants_size = engine->get_max_push_constants_size();
    const auto& device_info = engine->get_device_info();
    const auto input_count = params.input_layouts.size();
    const auto fused = get_supported_fused_eltwise_chain(node);
    const auto post_op = get_supported_fused_post_op(node);
    const auto base_input_count = is_unary_mode(node.as<eltwise>().get_primitive()->mode) ? 1U : 2U;
    OPENVINO_ASSERT(input_count == base_input_count + (fused.has_value() ? fused->size() : 0U),
                    "[GPU][Vulkan] Eltwise implementation creation received an unexpected input count");
    const auto& input0_layout = params.get_input_layout(0);
    const auto& input1_layout = input_count == 1 ? input0_layout : params.get_input_layout(1);
    kernel_kind kind = kernel_kind::broadcast_scalar;
    std::optional<scalar_constant> scalar;
    const auto fused_chain_length =
        fused.has_value() && fused->size() >= EltwiseMetadata::minimum_multi_stage_fused_chain_length ? checked_u32(fused->size(), "fused chain length") : 0U;
    if (post_op.has_value()) {
        kind = kernel_kind::fused_post_op;
    } else if (fused.has_value()) {
        kind = fused->size() == 1 ? (fused->front().broadcast_input ? kernel_kind::fused_broadcast : kernel_kind::fused_dense) : kernel_kind::fused_dense_chain;
        if (fused_chain_length != 0) {
            const auto& fused_input_layout = params.get_input_layout(fused->front().external_dependency_index);
            const auto vector_width =
                select_f32_dense_vector_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
            if (vector_width == dense_vector_width::vec4 && can_use_f32_no_tail_kernel(params.get_output_layout(0), vector_width, device_info)) {
                kind = kernel_kind::fused_dense_chain_f32_vec4_no_tail;
            }
        } else if (!fused->front().broadcast_input && max_push_constants_size >= EltwiseMetadata::dense_push_constant_bytes(true)) {
            const auto& fused_input_layout = params.get_input_layout(fused->front().external_dependency_index);
            const auto packed_width = select_packed_dense_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
            if (packed_width != packed_dense_width::none) {
                kind = select_packed_dense_kernel_kind(packed_width, params.get_output_layout(0).data_type, true);
            } else {
                const auto vector_width =
                    select_f32_dense_vector_width(input0_layout, input1_layout, params.get_output_layout(0), &fused_input_layout, device_info);
                const auto no_tail =
                    vector_width != dense_vector_width::scalar && can_use_f32_no_tail_kernel(params.get_output_layout(0), vector_width, device_info);
                kind = vector_width == dense_vector_width::scalar ? kernel_kind::fused_dense_push_constants
                                                                  : select_f32_vector_kernel_kind(vector_width, true, no_tail);
            }
        }
    } else if (is_unary_mode(node.as<eltwise>().get_primitive()->mode)) {
        kind = kernel_kind::unary;
    } else if (!params.is_dynamic()) {
        const auto& output_layout = params.get_output_layout(0);
        scalar = get_scalar_constant(node);
        if (scalar.has_value() && output_layout.count() >= broadcast_vector_min_elements) {
            kind = kernel_kind::scalar_constant;
        } else if (can_use_dense_kernel(input0_layout, input1_layout, output_layout) ||
                   can_use_packed_dense_kernel(input0_layout, input1_layout, output_layout, nullptr)) {
            kind = can_use_dense_kernel(input0_layout, input1_layout, output_layout) ? kernel_kind::dense : kernel_kind::broadcast_scalar;
            if (max_push_constants_size >= EltwiseMetadata::dense_push_constant_bytes(false)) {
                const auto packed_width = select_packed_dense_width(input0_layout, input1_layout, output_layout, nullptr, device_info);
                if (packed_width != packed_dense_width::none) {
                    kind = select_packed_dense_kernel_kind(packed_width, output_layout.data_type, false);
                } else if (can_use_dense_kernel(input0_layout, input1_layout, output_layout)) {
                    const auto vector_width = select_f32_dense_vector_width(input0_layout, input1_layout, output_layout, nullptr, device_info);
                    const auto no_tail = vector_width != dense_vector_width::scalar && can_use_f32_no_tail_kernel(output_layout, vector_width, device_info);
                    kind = vector_width == dense_vector_width::scalar ? kernel_kind::dense_push_constants
                                                                      : select_f32_vector_kernel_kind(vector_width, false, no_tail);
                } else {
                    kind = kernel_kind::broadcast_scalar;
                }
            }
        } else {
            const bool vector_kernel = can_use_broadcast_vector_kernel(output_layout);
            const auto provisional_kind = vector_kernel ? kernel_kind::broadcast_vector : kernel_kind::broadcast_scalar;
            const auto elements_per_invocation = select_generic_elements_per_invocation(provisional_kind,
                                                                                        input0_layout,
                                                                                        input1_layout,
                                                                                        output_layout,
                                                                                        nullptr,
                                                                                        node.as<eltwise>().get_primitive()->mode,
                                                                                        {},
                                                                                        device_info);
            const bool fast_kernel = should_use_fast_broadcast_kernel(input0_layout, input1_layout, output_layout, device_info, elements_per_invocation);
            kind = vector_kernel ? (fast_kernel ? kernel_kind::broadcast_fast_vector : kernel_kind::broadcast_vector)
                                 : (fast_kernel ? kernel_kind::broadcast_fast_scalar : kernel_kind::broadcast_scalar);
        }
        if (kind != kernel_kind::scalar_constant) {
            scalar.reset();
        }
        kind = select_pre_specialized_kernel_kind(kind, node.as<eltwise>().get_primitive()->mode, input0_layout, input1_layout, params.get_output_layout(0));
    }
    uint32_t elements_per_invocation = 0;
    if (!params.is_dynamic()) {
        const auto* fused_input_layout = fused.has_value() ? &params.get_input_layout(fused->front().external_dependency_index) : nullptr;
        std::vector<eltwise_mode> fused_modes;
        if (fused.has_value()) {
            fused_modes.reserve(fused->size());
            for (const auto& fused_stage : *fused) {
                fused_modes.push_back(fused_stage.descriptor->typed_desc<eltwise>()->mode);
            }
        }
        elements_per_invocation = select_generic_elements_per_invocation(kind,
                                                                         input0_layout,
                                                                         input1_layout,
                                                                         params.get_output_layout(0),
                                                                         fused_input_layout,
                                                                         node.as<eltwise>().get_primitive()->mode,
                                                                         fused_modes,
                                                                         device_info);
    }
    return std::make_unique<eltwise_impl>(kind, std::move(scalar), elements_per_invocation, fused_chain_length);
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::eltwise_impl)
