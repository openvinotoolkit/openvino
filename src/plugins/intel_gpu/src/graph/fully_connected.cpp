// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <algorithm>

#include "matmul_shape_inference.hpp"

namespace cldnn {
primitive_type_id fully_connected::type_id() {
    static primitive_type_base<fully_connected> instance;
    return &instance;
}

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

format::type get_preferred_format(const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout();

    // for 3d output we have to chose bfyx format
    if (impl_param.typed_desc<fully_connected>()->input_size == 3)
        return format::bfyx;

    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        (is_batch_after_spatial(input_layout.format.order()) ||
         input_layout.format == format::bs_x_bsv16 ||
         input_layout.format == format::bs_xs_xsv8_bsv8))
        return format::yxfb;

    bool no_spatial_padding = true;
    // C++ 11 range loop shouldn't be used here because of incorrect iterator functionality in mutable_array_ref<>
    for (size_t i = 0; i < input_layout.data_padding.lower_size().spatial.size(); ++i) {
        no_spatial_padding &= (input_layout.data_padding.lower_size().spatial[i] == 0);
    }
    for (size_t i = 0; i < input_layout.data_padding.upper_size().spatial.size(); ++i) {
        no_spatial_padding &= (input_layout.data_padding.upper_size().spatial[i] == 0);
    }

    if (input_layout.data_type == data_types::f32 &&
        input_layout.format == format::bfyx &&
        no_spatial_padding &&
        input_layout.batch() != 8)
        return format::bfyx;

    auto input_pitches = input_layout.get_pitches();
    if (input_layout.data_type == data_types::f16 &&
        input_layout.format == format::bfyx &&
        no_spatial_padding &&
        input_pitches.batch[0] % 2 == 0 &&
        input_layout.batch() != 16)
        return format::bfyx;

    // this condition tests whether our input is batch>1 in bfyx format, if yes there will be
    // extra reorder between input and this fc from bfyx to yxfb format (so
    // "is_batch_after_spatial" should return true)
    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        input_layout.format == format::bfyx &&
        input_layout.batch() > 1)
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
    auto output_type = input_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && desc->output_data_type)
        output_type = *desc->output_data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    auto reshape_to_2d = [](const ov::PartialShape& shape, int64_t feature) {
        auto staticShape = shape.to_shape();
        size_t total = std::accumulate(staticShape.begin(), staticShape.end(), 1, std::multiplies<size_t>());
        std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
        return reshapeSize;
    };

    int64_t feature = input_pshape[std::min(desc->input_size, static_cast<size_t>(4)) - 1].get_length();
    if (desc->input_size == 3) {
        feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
    }

    if (desc->input_size > 3) {
       input_layout.set_partial_shape(reshape_to_2d(input_pshape, feature));
    }
    if (weights_pshape.size() != 2) {
        weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
    }

    auto output_size = tensor(input_layout.batch(), weights_layout.batch(), 1, 1);
    if (desc->input_size == 3) {
        output_size = tensor(input_layout.batch(), input_layout.feature(), 1, weights_layout.batch());
    }
    format output_format = get_preferred_format(impl_param);

    return layout(output_type, output_format, output_size);
}

template<typename ShapeType>
std::vector<layout> fully_connected_inst::calc_output_layouts(fully_connected_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<fully_connected>();
    auto input_layout = impl_param.get_input_layout();
    auto weights_layout = *impl_param.weights_layout;

    auto default_out_dt = data_type_traits::is_floating_point(input_layout.data_type) ? input_layout.data_type : data_types::f32;
    auto output_type = desc->output_data_type.value_or(default_out_dt);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ov::op::v0::MatMul op;
    op.set_transpose_b(true);
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        weights_layout.get<ShapeType>()
    };

    ov::op::v0::shape_infer(&op, input_shapes, output_shapes);

    bool is_static = input_layout.is_static() && weights_layout.is_static();

    format::type output_format = is_static ? get_preferred_format(impl_param) :
                                             input_layout.format.value;

    return { layout{output_shapes[0], output_type, output_format} };
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

    node_info->add("fully connected info", fc_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fully_connected_inst::typed_primitive_inst(network& network, fully_connected_node const& node)
    : parent(network, node) { }
}  // namespace cldnn
