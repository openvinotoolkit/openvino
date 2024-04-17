// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_dot_product_attention_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <algorithm>
#include "utils.hpp"

#include "scaled_dot_product_attention_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scaled_dot_product_attention)

layout scaled_dot_product_attention_inst::calc_output_layout(scaled_dot_product_attention_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scaled_dot_product_attention>();

    // TBD: broadcasting
    auto input_layout = impl_param.get_input_layout(0);

    return layout {input_layout.data_type, format::bfyx, input_layout.get_tensor()};
}

template<typename ShapeType>
std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts(
    scaled_dot_product_attention_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.typed_desc<scaled_dot_product_attention>()->output_data_types[0]) == false &&
           "Output data type forcing is not supported for reshape_node!");
    // TBD: broadcasting
    auto desc = impl_param.typed_desc<scaled_dot_product_attention>();
    auto& input_layouts = impl_param.input_layouts;
    auto output_type = desc->output_data_types[0].value_or(input_layouts[0].data_type);

    ov::op::v13::ScaledDotProductAttention op;
    std::vector<ShapeType> input_shapes;
    std::transform(input_layouts.begin(), input_layouts.begin() + 3,
        std::back_inserter(input_shapes), [](const layout& l) {
            return l.get<ShapeType>();
        });

    std::vector<ShapeType> output_shapes = ov::op::v13::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], output_type, format::bfyx} };
}

template std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts<ov::PartialShape>(
    scaled_dot_product_attention_node const& node, const kernel_impl_params& impl_param);

std::string scaled_dot_product_attention_inst::to_string(scaled_dot_product_attention_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

scaled_dot_product_attention_inst::typed_primitive_inst(network& network, scaled_dot_product_attention_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
