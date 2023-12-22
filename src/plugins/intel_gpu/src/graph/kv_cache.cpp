// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/optionals.hpp"
#include "kv_cache_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(kv_cache)

kv_cache_inst::typed_primitive_inst(network& network, const kv_cache_node& node) :
    parent{network, node, false},
    memory_state::variable{node.get_primitive()->variable_info.variable_id} {
}

layout kv_cache_inst::calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> kv_cache_inst::calc_output_layouts(kv_cache_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<kv_cache>();
    auto output_data_type = desc->output_data_types[0].value_or(impl_param.get_input_layout().data_type);

    ov::intel_gpu::op::KVCache op;
    op.set_concat_axis(desc->concat_axis);
    op.set_gather_axis(desc->gather_axis);

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>(), impl_param.get_input_layout(1).get<ShapeType>()};

    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    return {layout({output_shapes[0], output_data_type, impl_param.get_output_layout().format})};
}

template std::vector<layout> kv_cache_inst::calc_output_layouts<ov::PartialShape>(kv_cache_node const& node, const kernel_impl_params& impl_param);

std::string kv_cache_inst::to_string(const kv_cache_node& node) {
    auto node_info = node.desc_to_json();
    json_composite kv_cache_info;
    kv_cache_info.add("input id", node.input().id());
    kv_cache_info.add("variable id", node.get_primitive()->variable_info.variable_id);
    kv_cache_info.add("variable shape", node.get_primitive()->variable_info.data_shape);
    kv_cache_info.add("variable type", node.get_primitive()->variable_info.data_type);
    kv_cache_info.add("concat axis", node.get_primitive()->concat_axis);
    kv_cache_info.add("gather axis", node.get_primitive()->gather_axis);
    node_info->add("kv_cache info", kv_cache_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
