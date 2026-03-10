// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <json_object.h>
#include <segment_max_inst.h>

#include <sstream>

#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(segment_max)

segment_max_inst::typed_primitive_inst(network& network, segment_max_node const& node) : parent(network, node) {}

layout segment_max_inst::calc_output_layout(segment_max_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> segment_max_inst::calc_output_layouts(segment_max_node const& node,
                                                          kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<segment_max>();

    auto input0_layout = impl_param.get_input_layout(0);

    const data_types output_type = impl_param.desc->output_data_types[0].value_or(input0_layout.data_type);

    // GPU-internal layouts are padded to 4D/5D (bfyx/bfzyx).  The core
    // SegmentMax shape_infer expects logical OV shapes and will reject the
    // padded segment_ids (e.g. [1,6,1,1] instead of [6]).  Compute the output
    // shape directly: output_shape = [num_segments] + data_shape[1:]
    auto output_pshape = input0_layout.get<ShapeType>();

    if (primitive->num_segments_val >= 0) {
        output_pshape[0] = primitive->num_segments_val;
    } else if (primitive->max_segment_id >= 0) {
        output_pshape[0] = primitive->max_segment_id + 1;
    } else {
        output_pshape[0] = ov::Dimension::dynamic();
    }

    return {layout{output_pshape, output_type, input0_layout.format}};
}

std::string segment_max_inst::to_string(segment_max_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite segment_max_info;
    segment_max_info.add("data id", node.input(0).id());
    segment_max_info.add("segment_ids id", node.input(1).id());
    segment_max_info.add("fill_mode", static_cast<int>(node.get_primitive()->fill_mode));
    node_info->add("segment_max info", segment_max_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
