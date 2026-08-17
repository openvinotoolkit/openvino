// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "activation_inst.h"
#include "backend_fusion_policy.hpp"
#include "eltwise_inst.h"
#include "eltwise_shader_abi.hpp"
#include "quantize_inst.h"
#include "reorder_inst.h"

namespace cldnn::vulkan {

class eltwise_fusion_policy final : public backend_fusion_policy {
public:
    bool limits_program_to_simple_fusions() const noexcept override {
        return true;
    }

    bool controls(fusion_kind kind) const noexcept override {
        return kind == fusion_kind::activation || kind == fusion_kind::quantize || kind == fusion_kind::eltwise || kind == fusion_kind::terminal_reorder;
    }

    bool can_consider_output(const program_node& node) const override {
        return node.is_type<eltwise>() || node.is_type<activation>() || node.is_type<quantize>() || node.is_type<reorder>();
    }

    fusion_decision evaluate(const fusion_query& query) const override {
        switch (query.kind) {
        case fusion_kind::activation:
            return evaluate_activation(query.producer, query.consumer);
        case fusion_kind::quantize:
            return evaluate_quantize(query.producer, query.consumer);
        case fusion_kind::eltwise:
            return query.peer == nullptr ? fusion_decision::reject : evaluate_eltwise(query.producer, *query.peer, query.consumer);
        case fusion_kind::terminal_reorder:
            return evaluate_terminal_reorder(query.producer, query.consumer);
        case fusion_kind::reorder_elimination:
            return evaluate_reorder_elimination(query.producer, query.consumer);
        }
        return fusion_decision::reject;
    }

    bool preserves_consumer_output(const program_node& consumer) const override {
        return consumer.is_output();
    }

private:
    static bool has_dense_storage(const layout& tensor_layout) {
        return !tensor_layout.is_dynamic() && !static_cast<bool>(tensor_layout.data_padding) &&
               tensor_layout.bytes_count() == tensor_layout.count() * data_type_traits::size_of(tensor_layout.data_type);
    }

    static bool has_same_dense_geometry(const layout& input_layout, const layout& output_layout) {
        return has_dense_storage(input_layout) && has_dense_storage(output_layout) && input_layout.get_shape() == output_layout.get_shape() &&
               input_layout.format == output_layout.format && input_layout.data_padding == output_layout.data_padding;
    }

    static bool has_enough_work(const program_node& node, const layout& output_layout) {
        if (output_layout.is_dynamic()) {
            return false;
        }
        const auto& info = node.get_program().get_engine().get_device_info();
        const auto subgroup_size = info.supported_simd_sizes.empty() ? 0 : info.supported_simd_sizes.front();
        return eltwise_shader_abi::post_op_fusion_has_enough_work(output_layout.count(), info.max_work_group_size, subgroup_size);
    }

    static bool supports_terminal_post_op(const program_node& producer, const layout& post_output_layout) {
        if (!producer.is_type<eltwise>() || producer.is_output() || producer.get_users().size() != 1 || producer.has_fused_primitives() ||
            producer.get_dependencies().empty() || producer.get_inputs_count() != 2) {
            return false;
        }
        const auto& base_output_layout = producer.get_output_layout();
        if ((base_output_layout.data_type != data_types::f16 && base_output_layout.data_type != data_types::f32) ||
            !has_same_dense_geometry(base_output_layout, post_output_layout) || !has_enough_work(producer, post_output_layout)) {
            return false;
        }
        return producer.get_input_layout(0).identical(base_output_layout) && producer.get_input_layout(1).identical(base_output_layout);
    }

    static bool supports_activation(activation_func function) {
        switch (function) {
        case activation_func::none:
        case activation_func::relu:
        case activation_func::relu_negative_slope:
        case activation_func::clamp:
        case activation_func::abs:
        case activation_func::linear:
        case activation_func::square:
        case activation_func::sqrt:
        case activation_func::negative:
        case activation_func::floor:
        case activation_func::ceil:
        case activation_func::reciprocal:
        case activation_func::hard_sigmoid:
        case activation_func::hsigmoid:
        case activation_func::hswish:
            return true;
        default:
            return false;
        }
    }

    static fusion_decision evaluate_activation(const program_node& producer, const program_node& consumer) {
        if (!consumer.is_type<activation>()) {
            return fusion_decision::reject;
        }
        const auto& activation_node = consumer.as<activation>();
        const auto desc = activation_node.get_primitive();
        const bool supported = activation_node.get_dependencies().size() == 1 && !desc->additional_params_input.is_valid() &&
                               supports_activation(desc->activation_function) &&
                               activation_node.get_output_layout().data_type == producer.get_output_layout().data_type &&
                               supports_terminal_post_op(producer, activation_node.get_output_layout());
        return supported ? fusion_decision::accept : fusion_decision::reject;
    }

    static fusion_decision evaluate_quantize(const program_node& producer, const program_node& consumer) {
        if (!consumer.is_type<quantize>()) {
            return fusion_decision::reject;
        }
        const auto& quantize_node = consumer.as<quantize>();
        const bool uses_output_range = quantize_node.get_per_tensor_output_range() && quantize_node.get_output_lo_val() < quantize_node.get_output_hi_val();
        const auto output_type = quantize_node.get_output_layout().data_type;
        const bool supported_output_type =
            output_type == data_types::f16 || output_type == data_types::f32 || output_type == data_types::i8 || output_type == data_types::u8;
        const bool supported = quantize_node.get_scale_shift_opt() && quantize_node.has_per_tensor_values() &&
                               (uses_output_range || !quantize_node.get_need_clamp()) && supported_output_type &&
                               supports_terminal_post_op(producer, quantize_node.get_output_layout());
        return supported ? fusion_decision::accept : fusion_decision::reject;
    }

    static bool is_numpy_broadcast_compatible(const layout& input_layout, const layout& target_layout) {
        const auto input_shape = input_layout.get_shape();
        const auto target_shape = target_layout.get_shape();
        if (input_shape.size() > target_shape.size()) {
            return false;
        }
        const auto leading_dimensions = target_shape.size() - input_shape.size();
        for (size_t input_axis = 0; input_axis < input_shape.size(); ++input_axis) {
            const auto input_dimension = input_shape[input_axis];
            const auto target_dimension = target_shape[leading_dimensions + input_axis];
            if (input_dimension != 1 && input_dimension != target_dimension) {
                return false;
            }
        }
        return true;
    }

    static bool supports_packed_output(const layout& output_layout) {
        if (output_layout.is_dynamic()) {
            return false;
        }
        const auto scalar_size = data_type_traits::size_of(output_layout.data_type);
        const auto packed_width = scalar_size < sizeof(uint32_t) ? sizeof(uint32_t) / scalar_size : 1;
        return scalar_size >= sizeof(uint32_t) || (output_layout.count() % packed_width == 0 && output_layout.get_linear_offset() % packed_width == 0);
    }

    static fusion_decision evaluate_eltwise(const program_node& producer, const program_node& peer, const program_node& consumer) {
        if (!consumer.is_type<eltwise>()) {
            return fusion_decision::reject;
        }
        const auto& consumer_primitive = consumer.as<eltwise>().get_primitive();
        const auto& producer_layout = producer.get_output_layout();
        const auto& peer_layout = peer.get_output_layout();
        const auto& output_layout = consumer.get_output_layout();
        const bool layouts_are_static = !producer_layout.is_dynamic() && !peer_layout.is_dynamic() && !output_layout.is_dynamic();
        const bool peer_is_broadcast = layouts_are_static && !peer_layout.identical(output_layout);
        const bool peer_layout_supported =
            layouts_are_static &&
            (!peer_is_broadcast || (consumer_primitive->broadcast_spec.m_type == ov::op::AutoBroadcastType::NUMPY &&
                                    data_type_traits::size_of(output_layout.data_type) >= sizeof(uint32_t) && producer.get_fused_primitives().empty() &&
                                    is_numpy_broadcast_compatible(peer_layout, output_layout)));
        const bool supported = !producer.is_output() && producer.is_type<eltwise>() &&
                               producer.get_fused_primitives().size() < eltwise_shader_abi::value(eltwise_shader_abi::limit::max_fused_chain_length) &&
                               producer.get_inputs_count() == 2 && layouts_are_static && has_dense_storage(producer_layout) && has_dense_storage(peer_layout) &&
                               has_dense_storage(output_layout) && supports_packed_output(output_layout) && peer_layout_supported &&
                               producer.get_input_layout(0).identical(output_layout) && producer.get_input_layout(1).identical(output_layout) &&
                               producer_layout.identical(output_layout);
        return supported ? fusion_decision::accept : fusion_decision::reject;
    }

    static fusion_decision evaluate_terminal_reorder(const program_node& producer, const program_node& consumer) {
        if (!consumer.is_type<reorder>()) {
            return fusion_decision::reject;
        }
        const auto& reorder_node = consumer.as<reorder>();
        const auto input_type = producer.get_output_layout().data_type;
        const auto output_type = reorder_node.get_output_layout().data_type;
        const bool supported = reorder_node.get_dependencies().size() == 1 && reorder_node.is_type_conversion_only() &&
                               (input_type == data_types::f16 || input_type == data_types::f32) &&
                               (output_type == data_types::f16 || output_type == data_types::f32) &&
                               supports_terminal_post_op(producer, reorder_node.get_output_layout());
        return supported ? fusion_decision::accept : fusion_decision::reject;
    }

    static fusion_decision evaluate_reorder_elimination(const program_node& producer, const program_node& consumer) {
        if (!producer.is_type<eltwise>()) {
            return fusion_decision::defer_to_common;
        }
        if (consumer.get_output_layout().is_dynamic()) {
            return fusion_decision::accept;
        }
        return has_enough_work(producer, consumer.get_output_layout()) ? fusion_decision::accept : fusion_decision::reject;
    }
};

}  // namespace cldnn::vulkan
