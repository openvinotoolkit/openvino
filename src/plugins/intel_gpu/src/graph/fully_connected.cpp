// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>
#include <algorithm>
#include "utils.hpp"
#include "swiglu_inst.h"

#include "matmul_shape_inference.hpp"
#include "glu_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(fully_connected)

namespace {
bool is_batch_after_spatial(const std::string order) {
    bool spatial_found = false;
    for (auto c : order) {
        switch (c) {
            case 'b':
            case 'n':
                return spatial_found;

            case 'x':
            case 'y':
            case 'z':
            case 'w':
            case 's':
                spatial_found = true;
                break;

            default:
                break;
        }
    }
    return false;
}

format::type get_preferred_format(fully_connected_node const& node, const kernel_impl_params& impl_param) {
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        return node.get_preferred_output_fmt();
    }

    auto input_layout = impl_param.get_input_layout();

    // for 3d output we have to chose bfyx format
    if (impl_param.typed_desc<fully_connected>()->input_size == 3)
        return format::bfyx;

    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        (is_batch_after_spatial(input_layout.format.order()) ||
         input_layout.format == format::bs_f_bsv16 ||
         input_layout.format == format::bs_fs_fsv8_bsv8))
        return format::yxfb;

    bool no_spatial_padding = true;
    // C++ 11 range loop shouldn't be used here because of incorrect iterator functionality in mutable_array_ref<>
    for (size_t i = 0; i < input_layout.get_spatial_rank(); ++i) {
        no_spatial_padding &= (input_layout.data_padding._lower_size[2 + i] == 0);
    }
    for (size_t i = 0; i < input_layout.get_spatial_rank(); ++i) {
        no_spatial_padding &= (input_layout.data_padding._upper_size[2 + i] == 0);
    }

    if (input_layout.data_type == data_types::f32 &&
        (input_layout.format == format::bfyx || input_layout.format == format::bfzyx) &&
        no_spatial_padding &&
        input_layout.batch() != 8)
        return input_layout.format;

    auto input_pitches = input_layout.get_pitches();
    if (input_layout.data_type == data_types::f16 &&
        (input_layout.format == format::bfyx || input_layout.format == format::bfzyx) &&
        no_spatial_padding &&
        input_pitches[0] % 2 == 0 &&
        input_layout.batch() != 16)
        return input_layout.format;

    // this condition tests whether our input is batch>1 in bfyx format, if yes there will be
    // extra reorder between input and this fc from bfyx to yxfb format (so
    // "is_batch_after_spatial" should return true)
    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        input_layout.format == format::bfyx &&
        input_layout.batch() > 1 &&
        input_pitches[2] == 1 &&
        input_pitches[3] == 1)
        return format::yxfb;

    return format::bfyx;
}

}  // namespace

layout fully_connected_inst::calc_output_layout(fully_connected_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<fully_connected>();

    auto input_layout = impl_param.get_input_layout();
    auto input_pshape = input_layout.get_partial_shape();
    auto weights_layout = *impl_param.weights_layout;
    auto weights_pshape = weights_layout.get_partial_shape();
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    if (data_type_traits::is_i8_u8(input_layout.data_type) && desc->output_data_types[0])
        output_type = *desc->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    const auto supports_immad = node.get_program().get_engine().get_device_info().supports_immad;

    auto reshape_to_2d = [](const ov::PartialShape& shape, int64_t feature) {
        auto staticShape = shape.to_shape();
        size_t total = std::accumulate(staticShape.begin(), staticShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
        return reshapeSize;
    };

    int64_t feature = input_pshape[std::min(desc->input_size, static_cast<size_t>(4)) - 1].get_length();

    if (desc->input_size == 3) {
        feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
    }

    if (weights_pshape.size() != 2) {
        weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
    }

    auto output_size = tensor();

    // If immad is supported, spatial dimensions are reshaped to 2d in order to select oneDnn impl,
    // because oneDnn doesn't support spatial dimensions for output.
    if (supports_immad) {
        if (desc->input_size > 3) {
            input_layout.set_partial_shape(reshape_to_2d(input_pshape, feature));
        }

        output_size = tensor(input_layout.batch(), weights_layout.batch(), 1, 1);
        if (desc->input_size == 3) {
            output_size = tensor(input_layout.batch(), input_layout.feature(), 1, weights_layout.batch());
        }
    } else {
        if (desc->input_size > 5) {
            input_layout.set_partial_shape(reshape_to_2d(input_pshape, feature));
        }

        output_size = tensor(input_layout.batch(), weights_layout.batch(), 1, 1);
        if (desc->input_size == 3) {
            output_size = tensor(input_layout.batch(), input_layout.feature(), 1, weights_layout.batch());
        } else if (desc->input_size == 4) {
            output_size = tensor(input_layout.batch(), input_layout.feature(), weights_layout.batch(), input_layout.spatial(1));
        } else if (desc->input_size == 5) {
            output_size = tensor(input_layout.batch(), input_layout.feature(), weights_layout.batch(), input_layout.spatial(1), input_layout.spatial(2));
        }
    }

    format output_format = get_preferred_format(node, impl_param);

    return layout(output_type, output_format, output_size);
}

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

    ov::op::v0::MatMul matmul_op;
    matmul_op.set_transpose_b(true);
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        weights_layout.get<ShapeType>()
    };

    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&matmul_op, input_shapes);
    bool has_swiglu = false;
    auto& fused_prims = node.get_fused_primitives();
    for (auto f : fused_prims) {
        if (f.is_type<swiglu>()) {
            has_swiglu = true;
            OPENVINO_ASSERT(fused_prims.size() == 1, "Other operation is fused in addition to swiglu!");
        }
    }
    if (has_swiglu) {
        ov::op::internal::GLU swiglu_op;
        OPENVINO_ASSERT(fused_prims.size() == 1);
        OPENVINO_ASSERT(fused_prims[0].typed_desc<swiglu>()->glu_type == ov::op::internal::GLU::GluType::Swish);
        swiglu_op.set_axis(fused_prims[0].typed_desc<swiglu>()->axis);
        swiglu_op.set_split_lengths(fused_prims[0].typed_desc<swiglu>()->split_lengths);
        swiglu_op.set_glu_type(fused_prims[0].typed_desc<swiglu>()->glu_type);
        std::vector<ShapeType> input_shapes = { output_shapes[0] };
        output_shapes = shape_infer(&swiglu_op, input_shapes);
    }

    bool is_static = input_layout.is_static() && weights_layout.is_static();
    bool allow_new_shape_infer = impl_param.get_program().is_new_shape_infer();
    format::type output_format = is_static && !allow_new_shape_infer ? get_preferred_format(node, impl_param) :
                                              input_layout.format.value;

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

    GPU_DEBUG_IF(orig_impl_param.get_program().get_config().get_disable_fake_alignment()) {
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
            auto is_8bit = weights_layout_dt == data_types::i8 || weights_layout_dt == data_types::u8;
            auto is_extra_alignment_needed = batch_size >= 256;
            fake_align_base = (is_4bit || is_8bit) && is_extra_alignment_needed ? 64 : 16;
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
