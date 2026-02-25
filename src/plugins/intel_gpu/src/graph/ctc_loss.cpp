// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>

#include "ctc_loss_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(ctc_loss)

template<typename ShapeType>
std::vector<layout> ctc_loss_inst::calc_output_layouts(ctc_loss_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& input_layout = impl_param.get_input_layout();
    return { layout{ ov::PartialShape{input_layout.get_partial_shape()[0]}, input_layout.data_type, input_layout.format} };
}

template std::vector<layout> ctc_loss_inst::calc_output_layouts<ov::PartialShape>(ctc_loss_node const& node, const kernel_impl_params& impl_param);


layout ctc_loss_inst::calc_output_layout(const ctc_loss_node& node, const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    std::vector<tensor::value_type> out_tensor = {input_layout.get_tensor().sizes().front(), 1, 1, 1};
    return {input_layout.data_type, input_layout.format, tensor(input_layout.format, out_tensor)};
}

std::string ctc_loss_inst::to_string(const ctc_loss_node& node) {
    auto primitive = node.get_primitive();
    json_composite ctc_loss_info;
    for (size_t i = 0; i < primitive->input_size(); ++i) {
        ctc_loss_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    ctc_loss_info.add("preprocess_collapse_repeated", primitive->preprocess_collapse_repeated);
    ctc_loss_info.add("ctc_merge_repeated", primitive->ctc_merge_repeated);
    ctc_loss_info.add("unique", primitive->unique);

    auto node_info = node.desc_to_json();
    node_info->add("ctc_loss info", ctc_loss_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
