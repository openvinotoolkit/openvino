// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(convert_color)

template<typename ShapeType>
std::vector<layout> convert_color_inst::calc_output_layouts(convert_color_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<convert_color>();

    auto src_fmt = desc->input_color_format;
    auto dst_fmt = desc->output_color_format;
    auto dst_is_rgb_or_bgr = dst_fmt == convert_color::color_format::BGR ||
                             dst_fmt == convert_color::color_format::RGB;
    auto inputs_count = desc->input_size();
    bool single_plane_input = inputs_count == 1;
    const size_t h_dim = 1;
    const size_t c_dim = 3;
    if ((src_fmt == convert_color::color_format::NV12 || src_fmt == convert_color::color_format::I420) && dst_is_rgb_or_bgr) {
        auto out_layout = impl_param.get_input_layout(0);
        out_layout.format = format::bfyx;
        auto out_shape = out_layout.get_partial_shape();
        out_shape[c_dim] = 3;
        if (single_plane_input) {
            out_shape[h_dim] = out_shape[h_dim] * 2 / 3;
        }
        out_layout.set_partial_shape(out_shape);

        return { out_layout };
    }
    OPENVINO_THROW("[GPU] Unsupported color format combinations");
}
template std::vector<layout> convert_color_inst::calc_output_layouts<ov::PartialShape>(convert_color_node const& node, const kernel_impl_params& impl_param);

layout convert_color_inst::calc_output_layout(convert_color_node const& /* node */, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<convert_color>();

    auto src_fmt = desc->input_color_format;
    auto dst_fmt = desc->output_color_format;
    auto dst_is_rgb_or_bgr = dst_fmt == convert_color::color_format::BGR ||
                             dst_fmt == convert_color::color_format::RGB;
    auto inputs_count = desc->input_size();
    bool single_plane_input = inputs_count == 1;
    const size_t h_dim = 1;
    const size_t c_dim = 3;
    if ((src_fmt == convert_color::color_format::NV12 || src_fmt == convert_color::color_format::I420) && dst_is_rgb_or_bgr) {
        auto out_layout = impl_param.get_input_layout(0);
        out_layout.format = format::bfyx;
        auto out_shape = out_layout.get_partial_shape();
        out_shape[c_dim] = 3;
        if (single_plane_input) {
            out_shape[h_dim] = out_shape[h_dim] * 2 / 3;
        }
        out_layout.set_partial_shape(out_shape);

        return out_layout;
    }
    OPENVINO_THROW("[GPU] Unsupported color format combinations");
}

std::string convert_color_inst::to_string(convert_color_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite convert_color_info;
    convert_color_info.add("input id", input.id());
    convert_color_info.add("memory type", desc->mem_type);
    convert_color_info.add("input color format", desc->input_color_format);
    convert_color_info.add("output color format", desc->output_color_format);

    node_info->add("convert_color info", convert_color_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

convert_color_inst::typed_primitive_inst(network& network, convert_color_node const& node) : parent(network, node) {}

}  // namespace cldnn
