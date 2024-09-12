// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>
#include <algorithm>
#include "utils.hpp"

#include "matmul_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(fully_connected)

template<typename ShapeType>
std::vector<layout> fully_connected_inst::calc_output_layouts(fully_connected_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<fully_connected>();
    auto input_layout = impl_param.get_input_layout();
    auto weights_layout = *impl_param.weights_layout;

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    if (data_type_traits::is_i8_u8(input_layout.data_type) && desc->output_data_types[0])
        output_type = *desc->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::op::v0::MatMul op;
    op.set_transpose_b(true);
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        weights_layout.get<ShapeType>()
    };

    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);

    format::type output_format = input_layout.format.value;

    if (node.get_preferred_output_fmt() != format::any)
        output_format = node.get_preferred_output_fmt();

    return { layout{output_shapes[0], output_type, output_format} };
}

kernel_impl_params fully_connected_inst::get_fake_aligned_params(kernel_impl_params const& orig_impl_param) {
    // fc_tiled_opt kernel is optimized for row shape aligned by 8.
    // Thus, use fake aligned shape at kernel execution for better performance.
    const auto& orig_input_layout = orig_impl_param.get_input_layout();
    const auto& orig_output_layout = orig_impl_param.get_output_layout();
    OPENVINO_ASSERT(orig_input_layout.is_static() && orig_output_layout.is_static(),
                    "in/out layouts should be static for fake alignment!");

    auto input_shape = orig_input_layout.get_partial_shape().to_shape();
    auto output_shape = orig_output_layout.get_partial_shape().to_shape();

    // Allow padding only for feature and outermost dimmension
    auto can_apply_fake_alignment = true;
    if (input_shape.size() == 3)
        can_apply_fake_alignment &= orig_input_layout.data_padding._lower_size[1] == 0 &&
                                    orig_input_layout.data_padding._upper_size[1] == 0;

    if (output_shape.size() == 3)
        can_apply_fake_alignment &= orig_output_layout.data_padding._lower_size[1] == 0 &&
                                    orig_output_layout.data_padding._upper_size[1] == 0;

    for (auto& fused_desc : orig_impl_param.fused_desc) {
        if (fused_desc.has_outer_dep()) {
            auto fused_op_input_layout = orig_impl_param.input_layouts[fused_desc.outer_dep_start_idx];
            // Check fused desc's input is still dynamic, then do not fake alignment
            if (fused_op_input_layout.is_dynamic()) {
                can_apply_fake_alignment = false;
                break;
            }
            // Check fused desc's input has full tensor, then do not fake alignment
            if (orig_output_layout.get_shape() == fused_op_input_layout.get_shape()) {
                can_apply_fake_alignment = false;
                break;
            }
        }
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_fake_alignment) {
        can_apply_fake_alignment = false;
    }

    if (orig_input_layout.format == format::bfyx && orig_output_layout.format == format::bfyx && can_apply_fake_alignment) {
        auto batch_size = std::accumulate(input_shape.begin(),
                                          input_shape.end() - 1,
                                          size_t{1},
                                          std::multiplies<size_t>());

        // Vector by matrix multiplication sometimes works slower if we align it
        if (batch_size == 1 && input_shape.back() >= 1024) {
            return std::move(orig_impl_param);
        }

        size_t fake_align_base = 8;
        if (orig_impl_param.dev_type == cldnn::device_type::integrated_gpu) {
            auto weights_layout_dt = orig_impl_param.weights_layout.value().data_type;
            auto is_4bit = weights_layout_dt == data_types::i4 || weights_layout_dt == data_types::u4;
            auto is_extra_alignment_needed = batch_size >= 256;
            fake_align_base = is_4bit && is_extra_alignment_needed ? 64 : 16;
        }

        std::fill(input_shape.begin(), input_shape.end() - 1, 1);
        std::fill(output_shape.begin(), output_shape.end() - 1, 1);

        input_shape[0] = align_to(batch_size, fake_align_base);
        output_shape[0] = align_to(batch_size, fake_align_base);

        auto updated_param = orig_impl_param;
        updated_param.input_layouts[0] = orig_input_layout.clone_with_other_shape(input_shape);
        updated_param.output_layouts[0] = orig_output_layout.clone_with_other_shape(output_shape);

        GPU_DEBUG_TRACE_DETAIL << "Apply fake alignment: input(" << orig_input_layout.to_short_string() << " -> "
                               << updated_param.input_layouts[0].to_short_string() << "), output("
                               << orig_output_layout.to_short_string() << " -> "
                               << updated_param.output_layouts[0].to_short_string() << ")\n";

        return updated_param;
    }
    return std::move(orig_impl_param);
}

template std::vector<layout> fully_connected_inst::calc_output_layouts<ov::PartialShape>(fully_connected_node const& node,
                                                                                         const kernel_impl_params& impl_param);

std::string fully_connected_inst::to_string(fully_connected_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto weights_id = desc->weights;

    std::stringstream primitive_description;

    json_composite fc_info;
    fc_info.add("weights id", weights_id);
    fc_info.add("bias id", bias_id);
    fc_info.add("compressed weights", desc->compressed_weights ? "true" : "false");
    if (desc->compressed_weights) {
        fc_info.add("decompression scale id", desc->decompression_scale);
        fc_info.add("decompression zp id", desc->decompression_zero_point);
        if (desc->decompression_zero_point_scalar.has_value()) {
            fc_info.add("decompression zp value", desc->decompression_zero_point_scalar.value());
        }
    }
    if (desc->dynamic_quantized_activation) {
        fc_info.add("activation scale id", desc->activation_scale.pid);
    }

    node_info->add("fully connected info", fc_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fully_connected_inst::typed_primitive_inst(network& network, fully_connected_node const& node)
    : parent(network, node) { }
}  // namespace cldnn
