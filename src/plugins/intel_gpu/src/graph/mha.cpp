// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(mha)

layout mha_inst::calc_output_layout(mha_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<mha>();

    auto input_layout = impl_param.get_input_layout(0);

    /* FIXME: add size assertion logic here */

    return input_layout;
}

template<typename ShapeType>
std::vector<layout> mha_inst::calc_output_layouts(mha_node const& node, kernel_impl_params const& impl_param) {
    /* UNIMPLEMENTED */
    OPENVINO_THROW("UNIMPLEMENTED error for MHA fusion on dynamic shape");
}

template std::vector<layout> mha_inst::calc_output_layouts<ov::PartialShape>(mha_node const& node, const kernel_impl_params& impl_param);

std::string mha_inst::to_string(mha_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

mha_inst::typed_primitive_inst(network& network, mha_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
