// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_inst.h"
#include "region_yolo_shape_inference.hpp"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(region_yolo)

layout region_yolo_inst::calc_output_layout(region_yolo_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for "
           "region_yolo_node!");
    auto input_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<region_yolo>();

    if (desc->do_softmax) {
        return cldnn::layout(
            input_layout.data_type,
            input_layout.format,
            tensor(input_layout.batch(),
                   input_layout.feature() * input_layout.spatial(0) * input_layout.spatial(1),
                   1,
                   1));
    } else {
        tensor::value_type features = (desc->classes + desc->coords + 1) * desc->mask_size;
        return cldnn::layout(
            input_layout.data_type,
            input_layout.format,
            tensor(input_layout.batch(), features, input_layout.spatial(0), input_layout.spatial(1)));
    }
}

template<typename ShapeType>
std::vector<layout> region_yolo_inst::calc_output_layouts(region_yolo_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<region_yolo>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    ov::op::v0::RegionYolo op;
    op.set_num_coords(static_cast<size_t>(desc->coords));
    op.set_num_classes(static_cast<size_t>(desc->classes));
    op.set_num_regions(static_cast<size_t>(desc->num));
    op.set_do_softmax(desc->do_softmax);
    op.set_mask(desc->mask);
    op.set_axis(desc->axis);
    op.set_end_axis(desc->end_axis);

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> region_yolo_inst::calc_output_layouts<ov::PartialShape>(region_yolo_node const& node, const kernel_impl_params& impl_param);

std::string region_yolo_inst::to_string(region_yolo_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto coords = desc->coords;
    auto classes = desc->classes;
    auto num = desc->num;
    auto do_softmax = desc->do_softmax;
    auto mask = desc->mask;
    auto mask_size = desc->mask_size;
    auto axis = desc->axis;
    auto end_axis = desc->end_axis;

    std::stringstream primitive_description;

    json_composite region_yolo_info;
    region_yolo_info.add("coords", coords);
    region_yolo_info.add("classes", classes);
    region_yolo_info.add("num", num);
    region_yolo_info.add("do_softmax", do_softmax);
    region_yolo_info.add("mask", mask);
    region_yolo_info.add("mask_size", mask_size);
    region_yolo_info.add("axis", axis);
    region_yolo_info.add("end_axis", end_axis);


    node_info->add("region yolo info", region_yolo_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

region_yolo_inst::typed_primitive_inst(network& network, region_yolo_node const& node) : parent(network, node) {}
}  // namespace cldnn
