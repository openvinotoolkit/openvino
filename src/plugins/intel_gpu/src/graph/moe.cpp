// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_inst.h"
#include "openvino/core/except.hpp"
#include "program_node.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "openvino/core/parallel.hpp"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe)

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout moe_inst::calc_output_layout(moe_node const& /* node */, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> moe_inst::calc_output_layouts(moe_node const& /* node */, kernel_impl_params const& impl_param) {
    return {impl_param.input_layouts[0]};
}

template std::vector<layout> moe_inst::calc_output_layouts<ov::PartialShape>(moe_node const& node, const kernel_impl_params& impl_param);

std::string moe_inst::to_string(moe_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite moe_info;

    node_info->add("moe info", moe_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
moe primitive is reusing memory with the input.
*/
moe_inst::typed_primitive_inst(network& network, moe_node const& node)
    : parent(network, node) {
}

}  // namespace cldnn
