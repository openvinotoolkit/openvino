// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "json_object.h"
#include "pa_kv_reorder_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(pa_kv_reorder)

layout pa_kv_reorder_inst::calc_output_layout(const pa_kv_reorder_node& node, const kernel_impl_params& impl_param) {
    (void)node;
    (void)impl_param;
    return {ov::PartialShape{1}, data_types::u8, format::bfyx};
}

template <typename ShapeType>
std::vector<layout> pa_kv_reorder_inst::calc_output_layouts(const pa_kv_reorder_node& node, const kernel_impl_params& impl_param) {
    (void)node;
    (void)impl_param;
    return {{ov::PartialShape{1}, data_types::u8, format::bfyx}};
}

template std::vector<layout> pa_kv_reorder_inst::calc_output_layouts<ov::PartialShape>(const pa_kv_reorder_node& node, const kernel_impl_params& impl_param);

std::string pa_kv_reorder_inst::to_string(const pa_kv_reorder_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

pa_kv_reorder_inst::typed_primitive_inst(network& network, const pa_kv_reorder_node& node) : parent(network, node) {}

void pa_kv_reorder_inst::on_execute() {}

void pa_kv_reorder_inst::update_output_memory() {
    return;
}

}  // namespace cldnn
