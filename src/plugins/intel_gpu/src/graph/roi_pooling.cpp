// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling_inst.h"
#include "roi_pooling_shape_inference.hpp"
#include "psroi_pooling_shape_inference.hpp"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(roi_pooling)

layout roi_pooling_inst::calc_output_layout(roi_pooling_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for roi_pooling_node!");
    auto desc = impl_param.typed_desc<roi_pooling>();
    layout data_layout = impl_param.get_input_layout(0);
    layout rois_layout = impl_param.get_input_layout(1);
    int num_rois = rois_layout.batch();
    int out_fm = desc->position_sensitive ? desc->output_dim : data_layout.feature();

    return layout(data_layout.data_type,
                  data_layout.format,
                  {num_rois, out_fm, desc->pooled_width, desc->pooled_height});
}

template<typename ShapeType>
std::vector<layout> roi_pooling_inst::calc_output_layouts(roi_pooling_node const& node, kernel_impl_params const& impl_param) {
 auto desc = impl_param.typed_desc<roi_pooling>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto output_type = desc->output_data_types[0].value_or(input0_layout.data_type);
    auto data_shape = input0_layout.get<ShapeType>();
    auto output_format = input0_layout.format;

    std::vector<ShapeType> output_shapes;
    ShapeType rois_shape = impl_param.get_input_layout(1).get<ShapeType>();
    if (desc->mode == cldnn::pooling_mode::deformable_bilinear) {
        auto group_size = desc->group_size;
        auto out_dim = desc->output_dim;
        auto num_rois = rois_shape[0];

        output_shapes = { ov::PartialShape{num_rois, out_dim, group_size, group_size} };
    } else if (desc->position_sensitive) {
        ov::op::v0::PSROIPooling op;
        op.set_spatial_scale(desc->spatial_scale);
        op.set_output_dim(desc->output_dim);
        op.set_group_size(desc->pooled_width);
        op.set_spatial_bins_x(desc->spatial_bins_x);
        op.set_spatial_bins_y(desc->spatial_bins_y);
        op.set_mode("average"); // mode doesn't matter

        std::vector<ShapeType> input_shapes = {
            data_shape,
            rois_shape
        };
        output_shapes = ov::op::v0::shape_infer(&op, input_shapes);
    } else {
        ov::op::v0::ROIPooling op;
        std::vector<int> output_size { desc->pooled_height, desc->pooled_width };
        op.set_output_roi({ output_size.begin(), output_size.end() });
        op.set_spatial_scale(desc->spatial_scale);

        std::vector<ShapeType> input_shapes = {
            data_shape,
            rois_shape
        };
        output_shapes = ov::op::v0::shape_infer(&op, input_shapes);
    }

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> roi_pooling_inst::calc_output_layouts<ov::PartialShape>(roi_pooling_node const& node, const kernel_impl_params& impl_param);

std::string roi_pooling_inst::to_string(roi_pooling_node const& node) {
    auto desc = node.get_primitive();
    auto mode = desc->mode == pooling_mode::max
                    ? "max"
                    : desc->mode == pooling_mode::bilinear
                          ? "bilinear"
                          : desc->mode == pooling_mode::deformable_bilinear ? "deformable_bilinear" : "average";
    auto is_ps = desc->position_sensitive ? "true" : "false";
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite roi_info;
    roi_info.add("mode", mode);
    roi_info.add("position sensitive", is_ps);
    roi_info.add("pooled_w", desc->pooled_width);
    roi_info.add("pooled_h", desc->pooled_height);
    roi_info.add("spatial_scale", desc->spatial_scale);
    roi_info.add("output_dim", desc->output_dim);
    roi_info.add("spatial_bins_x", desc->spatial_bins_x);
    roi_info.add("spatial_bins_y", desc->spatial_bins_y);
    roi_info.add("trans_std", desc->trans_std);
    roi_info.add("no_trans", desc->no_trans);
    roi_info.add("part_size", desc->part_size);

    node_info->add("roi info", roi_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
