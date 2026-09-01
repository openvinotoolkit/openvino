// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dispatch_builder.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "../eltwise_shader_abi.hpp"
#include "../shader_scalar_type.hpp"
#include "../vulkan_shader_abi.hpp"
#include "eltwise_inst.h"
#include "operation_semantics.hpp"
#include "vulkan/vulkan_memory.hpp"

namespace cldnn::vulkan::eltwise_detail {
namespace {

namespace shader_abi = eltwise_shader_abi;

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

}  // namespace

EltwiseDispatch EltwiseDispatch::build(eltwise_inst& instance,
                                       kernel_kind kind,
                                       const std::optional<scalar_constant>& scalar,
                                       const std::optional<fused_eltwise_chain>& fused,
                                       const std::optional<fused_post_op_info>& post_op,
                                       const EltwiseMetadata& metadata,
                                       memory::ptr metadata_memory,
                                       uint32_t elements_per_invocation,
                                       uint32_t local_size,
                                       bool restricted_kernel_available) {
    EltwiseDispatch result;
    auto& descriptor = result._descriptor;
    descriptor.layerID = instance.id();
    const auto element_count = metadata[shader_abi::index(shader_abi::metadata_field::element_count)];
    const auto desc = instance.get_typed_desc<eltwise>();
    const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
    const bool use_fused_kernel = is_fused_kernel(kind);
    const bool use_fused_post_op_kernel = is_fused_post_op_kernel(kind);
    const bool use_dense_kernel = is_plain_dense_kernel(kind) || use_fused_kernel || use_fused_post_op_kernel;
    const bool use_broadcast_vector_kernel = is_broadcast_vector_kernel(kind);
    const bool use_fast_broadcast_kernel = is_fast_broadcast_kernel(kind);
    const bool use_unary_kernel = kind == kernel_kind::unary;
    const bool use_scalar_constant_kernel = kind == kernel_kind::scalar_constant;
    const bool use_single_fused_kernel = is_single_fused_kernel(kind);
    const bool use_fused_chain_kernel = is_fused_chain_kernel(kind);
    const bool use_dense_push_constants = uses_dense_push_constants(kind);
    const auto invocation_count = (element_count + elements_per_invocation - 1) / elements_per_invocation;
    descriptor.workGroups.global = {invocation_count, 1, 1};
    descriptor.workGroups.local = {local_size, 1, 1};

    auto& specialization_constants = result._specialization_constants;
    specialization_constants = {
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
        specialization_constants.push_back(
            {shader_abi::index(shader_abi::specialization_id::fused_input_type), metadata.fused_stage_value(0, shader_abi::fused_metadata_field::input_type)});
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
        descriptor.scalars = metadata.make_dense_push_constants(is_single_fused_dense_kernel(kind));
    }

    descriptor.arguments = {{argument_desc::Types::INPUT, 0}};
    if (use_scalar_constant_kernel) {
        descriptor.arguments.front().index = scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::index(shader_abi::tensor_index::input1)
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

    auto& arguments = result._arguments;
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
        arguments.intermediates = {std::move(metadata_memory)};
    }

    std::vector<memory::cptr> read_memories;
    if (use_scalar_constant_kernel) {
        const auto tensor_input_index = scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::index(shader_abi::tensor_index::input1)
                                                                                                : shader_abi::index(shader_abi::tensor_index::input0);
        read_memories.push_back(arguments.inputs.at(tensor_input_index));
    } else {
        read_memories.insert(read_memories.end(), arguments.inputs.begin(), arguments.inputs.end());
    }
    read_memories.insert(read_memories.end(), arguments.fused_op_inputs.begin(), arguments.fused_op_inputs.end());
    read_memories.insert(read_memories.end(), arguments.intermediates.begin(), arguments.intermediates.end());
    result._kernel_index = restricted_kernel_available && output_is_disjoint_from_reads(arguments.outputs.front(), read_memories) ? 1 : 0;
    return result;
}

}  // namespace cldnn::vulkan::eltwise_detail
