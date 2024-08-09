// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/op/sync_tensor.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/optionals.hpp"
#include <sync_tensor_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(sync_tensor)

sync_tensor_inst::typed_primitive_inst(network& network, const sync_tensor_node& node) :
    parent(network, node, !node.can_be_optimized() && (node.get_output_layout().is_static() || node.get_output_layout().has_upper_bound())) {
}

layout sync_tensor_inst::calc_output_layout(const sync_tensor_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> sync_tensor_inst::calc_output_layouts(sync_tensor_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<sync_tensor>();
    ov::intel_gpu::op::SyncTensor op(impl_param.w_size);
    op.set_output_size(desc->num_outputs);

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>()};

    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < desc->num_outputs; i++) {
        auto out_type = impl_param.get_input_layout(0).data_type;
        out_layouts.push_back(layout(output_shapes[i], out_type, impl_param.get_output_layout(i).format));
    }

    return out_layouts;
}

template std::vector<layout> sync_tensor_inst::calc_output_layouts<ov::PartialShape>(sync_tensor_node const& node, const kernel_impl_params& impl_param);
std::string sync_tensor_inst::to_string(const sync_tensor_node& node) {
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void sync_tensor_inst::on_execute() {
    update_output_memory();
}

void sync_tensor_inst::update_output_memory() {
    if (!can_be_optimized()) {
        auto my_rank = get_impl_params()->w_rank;
        _outputs[my_rank] = input_memory_ptr();
        return;
    }
    // do nothing for now
}
} // namespace cldnn