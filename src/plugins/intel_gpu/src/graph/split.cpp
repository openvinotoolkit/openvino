// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id split::type_id() {
    static primitive_type_base<split> instance;
    return &instance;
}

layout split_inst::calc_output_layout(split_node const& node) {
    assert(!node.get_primitive()->output_data_types.empty() &&
           "Output data type forcing is not supported for split_node!");
    auto output_ids = node.get_primitive()->output_ids;
    auto output_offsets = node.get_primitive()->output_offsets;
    auto param_num = output_ids.size();
    auto input_sizes = node.get_dependency(0).first->get_non_padded_output_layout().size;
    tensor null_tensor { 0, 0, 0, 0 };

    // check if output_ids count equals output_offsets count
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Output_ids count",
                          param_num,
                          "output_offsets count",
                          output_offsets.size(),
                          "Output_ids count/ output_offsets count mismatch");

    for (decltype(param_num) i = 0; i < param_num; i++) {
        if (i != param_num - 1)
            // check if output offset sizes is less than next output offset sizes
            CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                                  "output_offsets",
                                                  output_offsets[i],
                                                  "next output_offsets",
                                                  output_offsets[i + 1],
                                                  "Output_offsets tensor/ next input output_offsets tensor mismatch");
        else
            // check if output offset sizes matches output offsets sizes
            CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                                  "Output_offsets",
                                                  output_offsets[i],
                                                  "input sizes",
                                                  input_sizes,
                                                  "Output_offsets tensor/ input tensor mismatch");

        // check if offsets do not extend input sizes and if match the output sizes
        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Output_offsets",
                                           output_offsets[i],
                                           "0 value",
                                           null_tensor,
                                           "Invalid output_offsets: dims cannot be less than 0");
    }

    return node.input().get_non_padded_output_layout();
}

std::string split_inst::to_string(split_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto output_ids = desc->output_ids;
    auto output_offsets = desc->output_offsets;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite split_info;
    split_info.add("input id", input.id());
    split_info.add("output ids count", output_ids.size());
    split_info.add("offset count", output_offsets.size());

    node_info->add("split info", split_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

split_inst::typed_primitive_inst(network& network, split_node const& node) : parent(network, node) {
    CLDNN_ERROR_MESSAGE(node.id(), "Split primitive instance should not be created!");
}

}  // namespace cldnn
