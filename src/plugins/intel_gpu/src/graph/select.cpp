// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "select_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id select::type_id() {
    static primitive_type_base<select> instance;
    return &instance;
}

layout select_inst::calc_output_layout(select_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for select_node!");

    auto output_layout = node.input(1).get_non_padded_output_layout();

    if (node.get_primitive()->broadcast_type == "numpy") {
        auto input1_size = node.input(1).get_output_layout().size;
        auto input2_size = node.input(2).get_output_layout().size;
        output_layout.size = tensor::max(input1_size, input2_size);
    }

    return output_layout;
}

std::string select_inst::to_string(select_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite select_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        select_info.add("input_" + std::to_string(i), node.input(i).id());
    }

    node_info->add("select info", select_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

select_inst::typed_primitive_inst(network& network, select_node const& node) : parent(network, node) {
    auto& deps = node.get_dependencies();

    CLDNN_ERROR_LESS_THAN(node.id(),
                                "Number of inputs",
                                deps.size(),
                                "Expected number of inputs",
                                3,
                                "");

    if (deps[1]->get_output_layout().size != cldnn::tensor(1))
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Mask format",
                              deps[0]->get_output_layout().format,
                              "Positive input format",
                              deps[1]->get_output_layout().format,
                              "");

    if (deps[2]->get_output_layout().size != cldnn::tensor(1))
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Mask format",
                              deps[0]->get_output_layout().format,
                              "Positive input format",
                              deps[2]->get_output_layout().format,
                              "");

    if (node.get_primitive()->broadcast_type == "none") {
        CLDNN_ERROR_LAYOUT_MISMATCH(node.id(),
                                "Positive input layout",
                                deps[1]->get_output_layout(),
                                "Negative input layout",
                                deps[2]->get_output_layout(),
                                "");

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Mask size",
                                deps[0]->get_output_layout().size,
                                "Positive input format",
                                deps[1]->get_output_layout().size,
                                "");
    } else if (node.get_primitive()->broadcast_type == "numpy") {
        if (deps[1]->get_output_layout().size != cldnn::tensor(1) && deps[2]->get_output_layout().size != cldnn::tensor(1))
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Positive input format",
                                  deps[1]->get_output_layout().format,
                                  "Negative input format",
                                  deps[2]->get_output_layout().format,
                                  "");

        CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                "Positive input data type",
                                deps[1]->get_output_layout().data_type,
                                "Negative input data type",
                                deps[2]->get_output_layout().data_type,
                                "");

        auto dep1_size = deps[1]->get_output_layout().size;
        auto dep2_size = deps[2]->get_output_layout().size;
        cldnn::tensor output_tensor = tensor::max(dep1_size, dep2_size);
        auto max_dim_count = output_tensor.raw.size();

        for (size_t i = 0; i < deps.size(); i++) {
            for (size_t d = 0; d < max_dim_count; d++) {
                auto current_dim = deps[i]->get_output_layout().size.raw[d];

                CLDNN_ERROR_BOOL(node.id(),
                                    "Sizes equal or broadcast is possible",
                                    !(current_dim == output_tensor.raw[d] || current_dim == 1),
                                    "Invalid input shapes");
            }
        }
    } else {
        CLDNN_ERROR_MESSAGE(node.id(), "Unsupported broadcast_type: " + node.get_primitive()->broadcast_type);
    }
}
}  // namespace cldnn
