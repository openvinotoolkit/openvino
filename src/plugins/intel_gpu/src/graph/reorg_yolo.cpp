// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo_inst.h"
#include "reorg_yolo_shape_inference.hpp"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(reorg_yolo)

layout reorg_yolo_inst::calc_output_layout(reorg_yolo_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for "
           "reorg_yolo_node!");
    auto input_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<reorg_yolo>();
    auto stride = desc->stride;

    cldnn::layout layoutTemp = cldnn::layout(input_layout.data_type,
                                             input_layout.format,
                                             tensor(input_layout.batch(),
                                                    input_layout.feature() * stride * stride,
                                                    input_layout.spatial(0) / stride,
                                                    input_layout.spatial(1) / stride));
    return layoutTemp;
}

template<typename ShapeType>
std::vector<layout> reorg_yolo_inst::calc_output_layouts(reorg_yolo_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<reorg_yolo>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    ov::op::v0::ReorgYolo op;
    op.set_strides(static_cast<size_t>(desc->stride));

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> reorg_yolo_inst::calc_output_layouts<ov::PartialShape>(reorg_yolo_node const& node, const kernel_impl_params& impl_param);

std::string reorg_yolo_inst::to_string(reorg_yolo_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto stride = desc->stride;

    std::stringstream primitive_description;

    json_composite reorg_yolo_info;
    reorg_yolo_info.add("stride", stride);

    node_info->add("reorg yolo info", reorg_yolo_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
reorg_yolo_inst::typed_primitive_inst(network& network, reorg_yolo_node const& node) : parent(network, node) {}
}  // namespace cldnn
