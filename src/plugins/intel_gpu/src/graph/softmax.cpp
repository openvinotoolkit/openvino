// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(softmax)

layout softmax_inst::calc_output_layout(softmax_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for softmax_node!");

    auto output_layout = impl_param.get_input_layout();

    if (impl_param.has_fused_primitives())
        output_layout.data_type = impl_param.get_output_element_type();

    return output_layout;
}

std::string softmax_inst::to_string(softmax_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite softmax_info;
    softmax_info.add("dimension", desc->dimension);

    node_info->add("softmax_info", softmax_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

softmax_inst::typed_primitive_inst(network& network, softmax_node const& node) : parent(network, node) {
    //    auto& input_offset  = arg.input_offset;
    //    auto& output_offset = arg.output_offset;
    //    auto& output_size   = arg.output_size;
    //
    //    auto& input_inst  = arg.input[0].primitive().as<const memory&>().argument;
    //    auto& output_inst = arg.output[0].as<const memory&>().argument;
    //    for (auto &x : input_offset.raw) if (x < 0) throw std::runtime_error("Softmax negative input offset.");
    //
    //    for(size_t i = 0; i < input_inst.size.raw.size(); ++i) {
    //        if( input_inst.size.raw[i] < output_size.raw[i] +  input_offset.raw[i]) throw std::runtime_error("Softmax
    //        input/output size does not match."); if(output_inst.size.raw[i] < output_size.raw[i] +
    //        output_offset.raw[i]) throw std::runtime_error("Softmax sizes too small.");
    //    }

    // auto& input_inst = network.get_topology()->get_primitives().at(desc->input()[0]);
    // if (input_inst->output_layout->size.format == cldnn::format::bfyx)
    //    if (input_inst->output_layout->size.spatial[0] != 1 || input_inst->output_layout->size.spatial[1] != 1)
    //        throw std::runtime_error("Softmax input has more than one dimension per batch");
}
}  // namespace cldnn
