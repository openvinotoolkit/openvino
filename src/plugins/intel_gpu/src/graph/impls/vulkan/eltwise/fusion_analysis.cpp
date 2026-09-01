// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusion_analysis.hpp"

#include <optional>

#include "../shader_scalar_type.hpp"
#include "activation_inst.h"
#include "kernel_selection.hpp"
#include "operation_semantics.hpp"
#include "program_node.h"
#include "quantize_inst.h"
#include "reorder_inst.h"

namespace cldnn::vulkan::eltwise_detail {
namespace {

namespace shader_abi = eltwise_shader_abi;

constexpr size_t maximum_supported_rank = 8;
constexpr size_t maximum_fused_chain_length = shader_abi::value(shader_abi::limit::max_fused_chain_length);

std::optional<shader_abi::post_activation> post_activation_code(activation_func function) {
    switch (function) {
    case activation_func::none:
        return shader_abi::post_activation::none;
    case activation_func::relu:
        return shader_abi::post_activation::relu;
    case activation_func::relu_negative_slope:
        return shader_abi::post_activation::relu_negative_slope;
    case activation_func::clamp:
        return shader_abi::post_activation::clamp;
    case activation_func::abs:
        return shader_abi::post_activation::abs;
    case activation_func::linear:
        return shader_abi::post_activation::linear;
    case activation_func::square:
        return shader_abi::post_activation::square;
    case activation_func::sqrt:
        return shader_abi::post_activation::sqrt;
    case activation_func::negative:
        return shader_abi::post_activation::negative;
    case activation_func::floor:
        return shader_abi::post_activation::floor;
    case activation_func::ceil:
        return shader_abi::post_activation::ceil;
    case activation_func::reciprocal:
        return shader_abi::post_activation::reciprocal;
    case activation_func::hard_sigmoid:
        return shader_abi::post_activation::hard_sigmoid;
    case activation_func::hsigmoid:
        return shader_abi::post_activation::hsigmoid;
    case activation_func::hswish:
        return shader_abi::post_activation::hswish;
    default:
        return std::nullopt;
    }
}

bool same_dense_geometry(const layout& input_layout, const layout& output_layout) {
    return !input_layout.is_dynamic() && !output_layout.is_dynamic() && has_dense_storage(input_layout) && has_dense_storage(output_layout) &&
           input_layout.get_shape() == output_layout.get_shape() && input_layout.format == output_layout.format &&
           input_layout.data_padding == output_layout.data_padding;
}

uint32_t post_quantize_flags(const QuantizeFuseParams& params) {
    uint32_t flags = 0;
    flags |= params._need_pre_shift ? shader_abi::value(shader_abi::post_quantize_flag::need_pre_shift) : 0U;
    flags |= params._need_post_scale ? shader_abi::value(shader_abi::post_quantize_flag::need_post_scale) : 0U;
    flags |= params._need_post_shift ? shader_abi::value(shader_abi::post_quantize_flag::need_post_shift) : 0U;
    flags |= params._need_clamp ? shader_abi::value(shader_abi::post_quantize_flag::need_clamp) : 0U;
    flags |= params._need_min_clamp ? shader_abi::value(shader_abi::post_quantize_flag::need_min_clamp) : 0U;
    flags |= params._need_max_clamp ? shader_abi::value(shader_abi::post_quantize_flag::need_max_clamp) : 0U;
    flags |= params._per_tensor_output_range && params._out_lo < params._out_hi ? shader_abi::value(shader_abi::post_quantize_flag::use_output_range) : 0U;
    return flags;
}

}  // namespace

std::optional<fused_post_op_info> get_supported_fused_post_op(const program_node& node) {
    const auto& fused_primitives = node.get_fused_primitives();
    if (fused_primitives.size() != 1) {
        return std::nullopt;
    }
    const auto& fused = fused_primitives.front();
    const auto& input_layout = fused.input_layout;
    const auto& output_layout = fused.output_layout;
    if (!fused.deps.empty() || fused.total_num_deps != 1 || !same_dense_geometry(input_layout, output_layout) ||
        !output_layout.identical(node.get_output_layout(0)) || !one_of(input_layout.data_type, {data_types::f16, data_types::f32}) ||
        !is_supported_shader_scalar_type(output_layout.data_type) || !is_supported_format(output_layout.format.value) ||
        output_layout.get_rank() > maximum_supported_rank) {
        return std::nullopt;
    }

    fused_post_op_info result;
    result.descriptor = &fused;
    if (fused.is_type<activation>()) {
        const auto params = fused.get_typed_fuse_params<ActivationFuseParams>();
        const auto activation = post_activation_code(params->_desc->activation_function);
        if (!activation.has_value() || params->_desc->additional_params_input.is_valid() || input_layout.data_type != output_layout.data_type) {
            return std::nullopt;
        }
        result.kind = shader_abi::post_op_kind::activation;
        result.activation = *activation;
        return result;
    }
    if (fused.is_type<quantize>()) {
        const auto params = fused.get_typed_fuse_params<QuantizeFuseParams>();
        const bool all_values_are_per_tensor = params->_per_tensor_input_range && params->_per_tensor_input_scale &&
                                               (!params->_need_pre_shift || params->_per_tensor_input_shift) && params->_per_tensor_output_range &&
                                               (!params->_need_post_scale || params->_per_tensor_output_scale) &&
                                               (!params->_need_post_shift || params->_per_tensor_output_shift);
        if (!params->_scale_shift_opt || !all_values_are_per_tensor ||
            !one_of(output_layout.data_type, {data_types::f16, data_types::f32, data_types::i8, data_types::u8})) {
            return std::nullopt;
        }
        result.kind = shader_abi::post_op_kind::quantize;
        result.quantize_flags = post_quantize_flags(*params);
        return result;
    }
    if (fused.is_type<reorder>()) {
        const auto desc = fused.typed_desc<reorder>();
        if (!desc->truncate || input_layout.data_type == output_layout.data_type || !one_of(output_layout.data_type, {data_types::f16, data_types::f32})) {
            return std::nullopt;
        }
        result.kind = shader_abi::post_op_kind::convert;
        return result;
    }
    return std::nullopt;
}

bool can_use_fused_post_op_kernel(const layout& input0_layout, const layout& input1_layout, const fused_post_op_info& post_op, const layout& output_layout) {
    const auto& base_output_layout = post_op.descriptor->input_layout;
    if (!input0_layout.identical(base_output_layout) || !input1_layout.identical(base_output_layout) ||
        !same_dense_geometry(base_output_layout, output_layout)) {
        return false;
    }
    const auto packed_width = output_elements_per_invocation(output_layout);
    return packed_width == 1 || output_layout.get_linear_offset() % packed_width == 0;
}

bool is_numpy_broadcast_compatible(const layout& input_layout, const layout& output_layout) {
    if (input_layout.is_dynamic() || output_layout.is_dynamic() || input_layout.get_rank() > output_layout.get_rank()) {
        return false;
    }
    const auto input_shape = input_layout.get_shape();
    const auto output_shape = output_layout.get_shape();
    const auto leading_dimensions = output_shape.size() - input_shape.size();
    for (size_t input_axis = 0; input_axis < input_shape.size(); ++input_axis) {
        const auto input_dimension = input_shape[input_axis];
        const auto output_dimension = output_shape[leading_dimensions + input_axis];
        if (input_dimension != 1 && input_dimension != output_dimension) {
            return false;
        }
    }
    return true;
}

std::optional<fused_eltwise_chain> get_supported_fused_eltwise_chain(const program_node& node) {
    const auto& fused_primitives = node.get_fused_primitives();
    if (fused_primitives.empty() || fused_primitives.size() > maximum_fused_chain_length) {
        return std::nullopt;
    }

    const auto& output_layout = node.get_output_layout(0);
    if (output_layout.is_dynamic() || !has_dense_storage(output_layout) || node.get_dependencies().size() != 2 + fused_primitives.size()) {
        return std::nullopt;
    }
    for (size_t input_index : {size_t{0}, size_t{1}}) {
        const auto& input_layout = node.get_input_layout(input_index);
        if (input_layout.is_dynamic() || !input_layout.identical(output_layout) || !has_dense_storage(input_layout)) {
            return std::nullopt;
        }
    }

    fused_eltwise_chain chain;
    chain.reserve(fused_primitives.size());
    for (size_t stage = 0; stage < fused_primitives.size(); ++stage) {
        const auto& fused = fused_primitives[stage];
        if (!fused.is_type<eltwise>()) {
            return std::nullopt;
        }
        const auto fused_desc = fused.typed_desc<eltwise>();
        const bool coefficients_supported = fused_desc->coefficients.empty() || (fused_desc->mode == eltwise_mode::sum && fused_desc->coefficients.size() == 2);
        if (!is_fused_mode(fused_desc->mode) || !fused_desc->stride.empty() || !coefficients_supported || fused.inputs.size() != 2 ||
            fused.total_num_deps != 2 || fused.deps.size() != 1 || !fused.has_outer_dep()) {
            return std::nullopt;
        }

        std::optional<size_t> external_position;
        std::optional<size_t> internal_position;
        for (size_t position = 0; position < fused.inputs.size(); ++position) {
            const auto& input = fused.inputs[position];
            if (input.m_type == FusedInputType::EXTERNAL) {
                if (external_position.has_value() || input.m_idx >= node.get_dependencies().size()) {
                    return std::nullopt;
                }
                external_position = position;
            } else if ((stage == 0 && (input.m_type != FusedInputType::ORIGINAL || input.m_idx != 0)) ||
                       (stage != 0 && (input.m_type != FusedInputType::INTERNAL || input.m_idx != stage - 1))) {
                return std::nullopt;
            }
            if (input.m_type != FusedInputType::EXTERNAL) {
                if (internal_position.has_value()) {
                    return std::nullopt;
                }
                internal_position = position;
            }
        }
        if (!external_position.has_value() || !internal_position.has_value()) {
            return std::nullopt;
        }

        const auto external_dependency_index = fused.inputs[*external_position].m_idx;
        const auto expected_external_dependency_index = 2 + stage;
        if (external_dependency_index != expected_external_dependency_index || external_dependency_index != static_cast<size_t>(fused.outer_dep_start_idx)) {
            return std::nullopt;
        }
        const auto& input_layout = node.get_input_layout(external_dependency_index);
        const bool broadcast_input = !input_layout.identical(output_layout);
        if (input_layout.is_dynamic() || !has_dense_storage(input_layout) || !is_supported_shader_scalar_type(input_layout.data_type) ||
            !is_supported_format(input_layout.format.value) || input_layout.get_rank() > maximum_supported_rank ||
            (broadcast_input && (fused_primitives.size() != 1 || data_type_traits::size_of(output_layout.data_type) < sizeof(uint32_t) ||
                                 !is_numpy_broadcast_compatible(input_layout, output_layout)))) {
            return std::nullopt;
        }
        if (!broadcast_input && !can_use_fused_dense_kernel(node.get_input_layout(0), node.get_input_layout(1), input_layout, output_layout)) {
            return std::nullopt;
        }

        chain.push_back(fused_eltwise_info{&fused,
                                           external_dependency_index,
                                           *external_position == 0 ? shader_abi::fused_input_position::lhs : shader_abi::fused_input_position::rhs,
                                           broadcast_input});
    }

    return chain;
}

}  // namespace cldnn::vulkan::eltwise_detail
