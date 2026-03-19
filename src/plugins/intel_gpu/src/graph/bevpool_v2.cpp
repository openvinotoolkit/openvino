// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <json_object.h>
#include <bevpool_v2_inst.h>

#include <sstream>
#include <type_traits>

#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(bevpool_v2)

bevpool_v2_inst::typed_primitive_inst(network& network, bevpool_v2_node const& node) : parent(network, node) {}

layout bevpool_v2_inst::calc_output_layout(bevpool_v2_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> bevpool_v2_inst::calc_output_layouts(bevpool_v2_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<bevpool_v2>();
    const auto& input_layout = impl_param.get_input_layout(0);
    const auto input_shape = input_layout.get<ShapeType>();

    ShapeType output_shape;
    if constexpr (std::is_same_v<ShapeType, ov::PartialShape>) {
        ov::Dimension batch = ov::Dimension::dynamic();
        if (input_shape.rank().is_static() && input_shape.rank().get_length() > 0) {
            batch = input_shape[0];
        }

        output_shape = ShapeType{batch,
                                 static_cast<int64_t>(primitive->output_channels),
                                 static_cast<int64_t>(primitive->feature_height),
                                 static_cast<int64_t>(primitive->feature_width)};
    } else {
        const auto batch = input_shape.empty() ? size_t{1} : static_cast<size_t>(input_shape[0]);
        output_shape = ShapeType{batch,
                                 static_cast<size_t>(primitive->output_channels),
                                 static_cast<size_t>(primitive->feature_height),
                                 static_cast<size_t>(primitive->feature_width)};
    }

    return {layout{output_shape, input_layout.data_type, input_layout.format}};
}

std::string bevpool_v2_inst::to_string(bevpool_v2_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite bevpool_v2_info;
    bevpool_v2_info.add("input", node.input(0).id());
    bevpool_v2_info.add("input_channels", node.get_primitive()->input_channels);
    bevpool_v2_info.add("output_channels", node.get_primitive()->output_channels);
    bevpool_v2_info.add("image_width", node.get_primitive()->image_width);
    bevpool_v2_info.add("image_height", node.get_primitive()->image_height);
    bevpool_v2_info.add("feature_width", node.get_primitive()->feature_width);
    bevpool_v2_info.add("feature_height", node.get_primitive()->feature_height);
    node_info->add("bevpool_v2 info", bevpool_v2_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
