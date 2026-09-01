// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata_builder.hpp"

#include <cstring>
#include <optional>

#include "../shader_scalar_type.hpp"
#include "activation_inst.h"
#include "openvino/core/except.hpp"
#include "operation_semantics.hpp"
#include "quantize_inst.h"

namespace cldnn::vulkan::eltwise_detail {
namespace {

namespace shader_abi = eltwise_shader_abi;

uint32_t float_bits(float value) {
    uint32_t result = 0;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void append_u32_scalar(scalars_desc& scalars, uint32_t value) {
    scalar_desc scalar{};
    scalar.t = scalar_desc::Types::UINT32;
    scalar.v.u32 = value;
    scalars.push_back(scalar);
}

}  // namespace

class EltwiseMetadataBuilder {
public:
    static EltwiseMetadata build(const eltwise_inst& instance,
                                 const std::optional<scalar_constant>& scalar,
                                 const std::optional<fused_eltwise_chain>& fused,
                                 const std::optional<fused_post_op_info>& post_op) {
        const auto desc = instance.get_typed_desc<eltwise>();
        const auto base_input_count = is_unary_mode(desc->mode) ? 1U : 2U;
        const auto fused_input_count = fused.has_value() ? fused->size() : 0U;
        OPENVINO_ASSERT(instance.inputs_memory_count() == base_input_count && instance.get_fused_mem_count() == fused_input_count,
                        "[GPU][Vulkan] Eltwise received an unexpected regular or fused input count");
        const auto& input0_layout = instance.get_input_layout(0);
        const auto& input1_layout = base_input_count == 1 ? input0_layout : instance.get_input_layout(1);
        const auto& output_layout = instance.get_output_layout(0);
        OPENVINO_ASSERT(!input0_layout.is_dynamic() && !input1_layout.is_dynamic() && !output_layout.is_dynamic(),
                        "[GPU][Vulkan] Eltwise execution requires resolved runtime layouts");
        const auto output_rank = checked_u32(output_layout.get_shape().size(), "output rank");
        OPENVINO_ASSERT(output_rank <= EltwiseMetadata::maximum_rank, "[GPU][Vulkan] Eltwise supports tensors with rank up to ", EltwiseMetadata::maximum_rank);

        EltwiseMetadata result;
        auto& words = result._words;
        words[shader_abi::index(shader_abi::metadata_field::element_count)] = checked_u32(output_layout.count(), "element count");
        words[shader_abi::index(shader_abi::metadata_field::mode)] = shader_abi::value(shader_mode_code(desc->mode));
        words[shader_abi::index(shader_abi::metadata_field::rank)] = output_rank;
        const auto scalar_tensor_index =
            scalar.has_value() && scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::tensor_index::input1 : shader_abi::tensor_index::input0;
        const auto& scalar_tensor_layout = scalar_tensor_index == shader_abi::tensor_index::input0 ? input0_layout : input1_layout;
        words[shader_abi::index(shader_abi::metadata_field::flags)] =
            ((scalar.has_value() ? can_use_scalar_linear_storage(scalar_tensor_layout, output_layout)
                                 : can_use_linear_storage(input0_layout, input1_layout, output_layout))
                 ? shader_abi::value(shader_abi::storage_flag::linear)
                 : 0U) |
            (output_elements_per_invocation(output_layout) > 1 ? shader_abi::value(shader_abi::storage_flag::packed_output) : 0U) |
            (scalar.has_value() && scalar->input_index == shader_abi::tensor_index::input0 ? shader_abi::value(shader_abi::storage_flag::scalar_constant_input0)
                                                                                           : 0U) |
            (fused.has_value() && fused->size() == 1 && fused->front().broadcast_input &&
                     instance.get_input_layout(fused->front().external_dependency_index).count() == 1
                 ? shader_abi::value(shader_abi::storage_flag::fused_scalar_input)
                 : 0U);
        words[shader_abi::index(shader_abi::metadata_field::input0_type)] = shader_abi::value(to_shader_scalar_type(input0_layout.data_type));
        words[shader_abi::index(shader_abi::metadata_field::input1_type)] = shader_abi::value(to_shader_scalar_type(input1_layout.data_type));
        words[shader_abi::index(shader_abi::metadata_field::output_type)] = shader_abi::value(to_shader_scalar_type(output_layout.data_type));
        words[shader_abi::index(shader_abi::metadata_field::python_division)] = desc->m_pythondiv ? 1U : 0U;
        words[shader_abi::index(shader_abi::metadata_field::infinity_detection)] =
            desc->mode == eltwise_mode::is_inf && desc->coefficients.size() == 2
                ? (desc->coefficients[0] != 0.0f ? shader_abi::value(shader_abi::infinity_flag::negative) : 0U) |
                      (desc->coefficients[1] != 0.0f ? shader_abi::value(shader_abi::infinity_flag::positive) : 0U)
                : shader_abi::all_infinities_mask;
        words[shader_abi::index(shader_abi::metadata_field::input_count)] = base_input_count;
        words[shader_abi::index(shader_abi::metadata_field::input0_coefficient)] = float_bits(desc->coefficients.empty() ? 1.0f : desc->coefficients[0]);
        words[shader_abi::index(shader_abi::metadata_field::input1_coefficient)] = float_bits(desc->coefficients.empty() ? 1.0f : desc->coefficients[1]);
        write_tensor(words, shader_abi::tensor_index::input0, input0_layout, output_rank);
        write_tensor(words, shader_abi::tensor_index::input1, input1_layout, output_rank);
        write_tensor(words, shader_abi::tensor_index::output, output_layout, output_rank);
        std::optional<uint32_t> extra_tensor_base;
        if (fused.has_value() && fused->size() == 1 && fused->front().broadcast_input) {
            write_tensor_at(words,
                            EltwiseMetadata::fused_broadcast_metadata_base,
                            instance.get_input_layout(fused->front().external_dependency_index),
                            output_rank);
            extra_tensor_base = EltwiseMetadata::fused_broadcast_metadata_base;
        }
        words[shader_abi::index(shader_abi::metadata_field::rank)] = collapse_dimensions(words, output_rank, extra_tensor_base);
        write_fast_divisors(words, fused.has_value() && fused->size() > 1);
        if (scalar.has_value()) {
            const auto scalar_metadata_base = EltwiseMetadata::header_words + shader_abi::index(scalar->input_index) * EltwiseMetadata::tensor_words;
            words[scalar_metadata_base] = scalar->bits[0];
            words[scalar_metadata_base + EltwiseMetadata::maximum_rank] = scalar->bits[1];
        }
        if (fused.has_value()) {
            words[EltwiseMetadata::fused_chain_metadata_base + shader_abi::index(shader_abi::fused_chain_metadata_field::chain_length)] =
                checked_u32(fused->size(), "fused Eltwise chain length");
            for (size_t stage = 0; stage < fused->size(); ++stage) {
                const auto& fused_stage = fused->at(stage);
                const auto fused_desc = fused_stage.descriptor->typed_desc<eltwise>();
                const auto& fused_input_layout = instance.get_input_layout(fused_stage.external_dependency_index);
                const auto stage_metadata_base = EltwiseMetadata::fused_metadata_base + stage * EltwiseMetadata::fused_metadata_words;
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::mode)] = shader_abi::value(shader_mode_code(fused_desc->mode));
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_type)] =
                    shader_abi::value(to_shader_scalar_type(fused_input_layout.data_type));
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_position)] =
                    shader_abi::value(fused_stage.external_position);
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::python_division)] = fused_desc->m_pythondiv ? 1U : 0U;
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input0_coefficient)] =
                    float_bits(fused_desc->coefficients.empty() ? 1.0f : fused_desc->coefficients[0]);
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input1_coefficient)] =
                    float_bits(fused_desc->coefficients.empty() ? 1.0f : fused_desc->coefficients[1]);
                words[stage_metadata_base + shader_abi::index(shader_abi::fused_metadata_field::input_offset)] =
                    checked_u32(fused_input_layout.get_linear_offset(), "fused input base offset");
            }
        }
        if (post_op.has_value()) {
            const auto write_post_value = [&](shader_abi::post_op_metadata_field field, float value) {
                words[EltwiseMetadata::post_op_metadata_base + shader_abi::index(field)] = float_bits(value);
            };
            if (post_op->kind == shader_abi::post_op_kind::activation) {
                const auto params = post_op->descriptor->get_typed_fuse_params<ActivationFuseParams>();
                write_post_value(shader_abi::post_op_metadata_field::parameter_a, params->_desc->additional_params.a);
                write_post_value(shader_abi::post_op_metadata_field::parameter_b, params->_desc->additional_params.b);
            } else if (post_op->kind == shader_abi::post_op_kind::quantize) {
                const auto params = post_op->descriptor->get_typed_fuse_params<QuantizeFuseParams>();
                write_post_value(shader_abi::post_op_metadata_field::input_low, params->_in_lo);
                write_post_value(shader_abi::post_op_metadata_field::input_high, params->_in_hi);
                write_post_value(shader_abi::post_op_metadata_field::input_scale, params->_in_scale);
                write_post_value(shader_abi::post_op_metadata_field::input_shift, params->_in_shift);
                write_post_value(shader_abi::post_op_metadata_field::output_low, params->_out_lo);
                write_post_value(shader_abi::post_op_metadata_field::output_high, params->_out_hi);
                write_post_value(shader_abi::post_op_metadata_field::output_scale, params->_out_scale);
                write_post_value(shader_abi::post_op_metadata_field::output_shift, params->_out_shift);
            }
        }
        return result;
    }

    static uint32_t collapsed_broadcast_rank(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
        const auto output_rank = checked_u32(output_layout.get_shape().size(), "output rank");
        EltwiseMetadata::storage_type words{};
        write_tensor(words, shader_abi::tensor_index::input0, input0_layout, output_rank);
        write_tensor(words, shader_abi::tensor_index::input1, input1_layout, output_rank);
        write_tensor(words, shader_abi::tensor_index::output, output_layout, output_rank);
        return collapse_dimensions(words, output_rank);
    }

private:
    static void write_tensor_at(EltwiseMetadata::storage_type& words, uint32_t base, const layout& tensor_layout, uint32_t output_rank) {
        const auto shape = tensor_layout.get_shape();
        const auto pitches = tensor_layout.get_pitches();
        OPENVINO_ASSERT(shape.size() <= pitches.size() && shape.size() <= output_rank, "[GPU][Vulkan] Eltwise received an invalid tensor rank");

        const auto leading_dimensions = output_rank - checked_u32(shape.size(), "rank");
        for (uint32_t axis = 0; axis < EltwiseMetadata::maximum_rank; ++axis) {
            words[base + axis] = 1;
            words[base + EltwiseMetadata::maximum_rank + axis] = 0;
        }
        for (uint32_t axis = 0; axis < shape.size(); ++axis) {
            const auto output_axis = leading_dimensions + axis;
            words[base + output_axis] = checked_u32(shape[axis], "dimension");
            words[base + EltwiseMetadata::maximum_rank + output_axis] = checked_u32(pitches[axis], "pitch");
        }
        words[base + EltwiseMetadata::maximum_rank * 2] = checked_u32(tensor_layout.get_linear_offset(), "base offset");
    }

    static void write_tensor(EltwiseMetadata::storage_type& words, shader_abi::tensor_index tensor, const layout& tensor_layout, uint32_t output_rank) {
        write_tensor_at(words, EltwiseMetadata::header_words + shader_abi::index(tensor) * EltwiseMetadata::tensor_words, tensor_layout, output_rank);
    }

    static bool collapse_tensor_axis(const EltwiseMetadata::storage_type& words,
                                     uint32_t base,
                                     uint32_t axis,
                                     uint32_t output_left_dimension,
                                     uint32_t output_right_dimension,
                                     uint32_t& collapsed_dimension,
                                     uint32_t& collapsed_pitch) {
        const auto left_dimension = words[base + axis];
        const auto right_dimension = words[base + axis + 1];
        const auto left_pitch = words[base + EltwiseMetadata::maximum_rank + axis];
        const auto right_pitch = words[base + EltwiseMetadata::maximum_rank + axis + 1];

        if (output_left_dimension == 1) {
            collapsed_dimension = right_dimension;
            collapsed_pitch = right_pitch;
            return true;
        }
        if (output_right_dimension == 1) {
            collapsed_dimension = left_dimension;
            collapsed_pitch = left_pitch;
            return true;
        }
        if (left_dimension == 1 && right_dimension == 1) {
            collapsed_dimension = 1;
            collapsed_pitch = 0;
            return true;
        }
        if (left_dimension == output_left_dimension && right_dimension == output_right_dimension &&
            static_cast<uint64_t>(left_pitch) == static_cast<uint64_t>(right_pitch) * right_dimension) {
            collapsed_dimension = checked_u32(static_cast<uint64_t>(left_dimension) * right_dimension, "collapsed dimension");
            collapsed_pitch = right_pitch;
            return true;
        }
        return false;
    }

    static uint32_t collapse_dimensions(EltwiseMetadata::storage_type& words, uint32_t rank, std::optional<uint32_t> extra_tensor_base = std::nullopt) {
        if (rank < 2) {
            return rank;
        }

        for (int32_t axis = static_cast<int32_t>(rank) - 2; axis >= 0; --axis) {
            const auto output_base = EltwiseMetadata::header_words + shader_abi::index(shader_abi::tensor_index::output) * EltwiseMetadata::tensor_words;
            const auto output_left_dimension = words[output_base + static_cast<uint32_t>(axis)];
            const auto output_right_dimension = words[output_base + static_cast<uint32_t>(axis) + 1];
            std::array<uint32_t, EltwiseMetadata::tensor_count + 1> tensor_bases{};
            for (uint32_t tensor = 0; tensor < EltwiseMetadata::tensor_count; ++tensor) {
                tensor_bases[tensor] = EltwiseMetadata::header_words + tensor * EltwiseMetadata::tensor_words;
            }
            const auto active_tensor_count = EltwiseMetadata::tensor_count + (extra_tensor_base.has_value() ? 1U : 0U);
            if (extra_tensor_base.has_value()) {
                tensor_bases[EltwiseMetadata::tensor_count] = *extra_tensor_base;
            }
            std::array<uint32_t, EltwiseMetadata::tensor_count + 1> collapsed_dimensions{};
            std::array<uint32_t, EltwiseMetadata::tensor_count + 1> collapsed_pitches{};
            bool can_collapse = true;
            for (uint32_t tensor = 0; tensor < active_tensor_count; ++tensor) {
                can_collapse &= collapse_tensor_axis(words,
                                                     tensor_bases[tensor],
                                                     static_cast<uint32_t>(axis),
                                                     output_left_dimension,
                                                     output_right_dimension,
                                                     collapsed_dimensions[tensor],
                                                     collapsed_pitches[tensor]);
            }
            if (!can_collapse) {
                continue;
            }

            for (uint32_t tensor = 0; tensor < active_tensor_count; ++tensor) {
                const auto base = tensor_bases[tensor];
                words[base + static_cast<uint32_t>(axis)] = collapsed_dimensions[tensor];
                words[base + EltwiseMetadata::maximum_rank + static_cast<uint32_t>(axis)] = collapsed_pitches[tensor];
                for (uint32_t shifted_axis = static_cast<uint32_t>(axis) + 1; shifted_axis + 1 < rank; ++shifted_axis) {
                    words[base + shifted_axis] = words[base + shifted_axis + 1];
                    words[base + EltwiseMetadata::maximum_rank + shifted_axis] = words[base + EltwiseMetadata::maximum_rank + shifted_axis + 1];
                }
                words[base + rank - 1] = 1;
                words[base + EltwiseMetadata::maximum_rank + rank - 1] = 0;
            }
            --rank;
        }
        return rank;
    }

    static uint32_t fast_divisor_reciprocal(uint32_t divisor) {
        OPENVINO_ASSERT(divisor != 0, "[GPU][Vulkan] Eltwise broadcast divisor must be non-zero");
        if (divisor == 1) {
            return 0;
        }
        return static_cast<uint32_t>((uint64_t{1} << 32) / divisor);
    }

    static void write_fast_divisors(EltwiseMetadata::storage_type& words, bool fused_chain) {
        const auto rank = words[shader_abi::index(shader_abi::metadata_field::rank)];
        const auto output_base = EltwiseMetadata::header_words + shader_abi::index(shader_abi::tensor_index::output) * EltwiseMetadata::tensor_words;
        const auto metadata_base = fused_chain ? EltwiseMetadata::fused_chain_fast_divisor_metadata_base : EltwiseMetadata::regular_fast_divisor_metadata_base;
        for (uint32_t axis = 0; axis < EltwiseMetadata::maximum_rank; ++axis) {
            const auto dimension = axis < rank ? words[output_base + axis] : 1U;
            words[metadata_base + axis] = fast_divisor_reciprocal(dimension);
        }
    }
};

EltwiseMetadata EltwiseMetadata::build(const eltwise_inst& instance,
                                       const std::optional<scalar_constant>& scalar,
                                       const std::optional<fused_eltwise_chain>& fused,
                                       const std::optional<fused_post_op_info>& post_op) {
    return EltwiseMetadataBuilder::build(instance, scalar, fused, post_op);
}

uint32_t EltwiseMetadata::buffer_word_count(kernel_kind kind) {
    if (is_fused_post_op_kernel(kind)) {
        return post_op_metadata_words;
    }
    if (is_fused_broadcast_kernel(kind)) {
        return fused_broadcast_metadata_words;
    }
    if (is_fused_chain_kernel(kind)) {
        return fused_chain_metadata_words;
    }
    return regular_metadata_words;
}

uint32_t EltwiseMetadata::collapsed_broadcast_rank(const layout& input0_layout, const layout& input1_layout, const layout& output_layout) {
    return EltwiseMetadataBuilder::collapsed_broadcast_rank(input0_layout, input1_layout, output_layout);
}

uint32_t EltwiseMetadata::active_broadcast_axes(shader_abi::tensor_index tensor) const {
    const auto rank = _words[shader_abi::index(shader_abi::metadata_field::rank)];
    const auto base = header_words + shader_abi::index(tensor) * tensor_words;
    uint32_t axes = 0;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (_words[base + axis] != 1 && _words[base + maximum_rank + axis] != 0) {
            axes |= 1U << axis;
        }
    }
    return axes;
}

uint32_t EltwiseMetadata::fused_stage_value(size_t stage, shader_abi::fused_metadata_field field) const {
    return _words[fused_metadata_base + stage * fused_metadata_words + shader_abi::index(field)];
}

scalars_desc EltwiseMetadata::make_dense_push_constants(bool fused) const {
    constexpr auto dense_words = shader_abi::index(shader_abi::dense_metadata_field::count);
    constexpr auto fused_dense_words = dense_words + shader_abi::index(shader_abi::fused_dense_metadata_field::count);
    scalars_desc result;
    result.reserve(fused ? fused_dense_words : dense_words);
    const auto tensor_offset = [&](shader_abi::tensor_index tensor) {
        return _words[header_words + shader_abi::index(tensor) * tensor_words + maximum_rank * 2];
    };
    append_u32_scalar(result, _words[shader_abi::index(shader_abi::metadata_field::element_count)]);
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::input0));
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::input1));
    append_u32_scalar(result, tensor_offset(shader_abi::tensor_index::output));
    append_u32_scalar(result, _words[shader_abi::index(shader_abi::metadata_field::python_division)]);
    append_u32_scalar(result, _words[shader_abi::index(shader_abi::metadata_field::infinity_detection)]);
    append_u32_scalar(result, _words[shader_abi::index(shader_abi::metadata_field::input0_coefficient)]);
    append_u32_scalar(result, _words[shader_abi::index(shader_abi::metadata_field::input1_coefficient)]);
    if (fused) {
        append_u32_scalar(result, fused_stage_value(0, shader_abi::fused_metadata_field::input_offset));
        append_u32_scalar(result, fused_stage_value(0, shader_abi::fused_metadata_field::python_division));
        append_u32_scalar(result, fused_stage_value(0, shader_abi::fused_metadata_field::input0_coefficient));
        append_u32_scalar(result, fused_stage_value(0, shader_abi::fused_metadata_field::input1_coefficient));
    }
    return result;
}

}  // namespace cldnn::vulkan::eltwise_detail
