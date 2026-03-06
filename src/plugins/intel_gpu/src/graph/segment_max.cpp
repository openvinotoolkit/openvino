// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <json_object.h>
#include <segment_max_inst.h>

#include <sstream>

#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"
#include "segment_max_shape_inference.hpp"

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

    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    TensorsContainer const_data(&impl_param.get_stream(), impl_param.memory_deps);

    ov::op::v16::SegmentMax op;
    auto output_shapes = ov::op::v16::shape_infer(&op, input_shapes, cldnn::make_tensor_accessor(const_data));

    // shape_infer with a default-constructed op cannot detect 3-input form
    // (op->inputs().size() is 0), so apply stored num_segments if available.
    if (primitive->num_segments_val >= 0) {
        output_shapes[0][0] = primitive->num_segments_val;
    }

    return {layout{output_shapes[0], output_type, input0_layout.format}};
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
