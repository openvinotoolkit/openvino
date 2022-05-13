// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {

primitive_type_id reshape::type_id() {
    static primitive_type_base<reshape> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto input_layout = node.input().get_non_padded_output_layout();
    auto sizes = node.get_primitive()->output_shape.sizes();
    auto input_sizes = input_layout.size.sizes();
    size_t need_recalc = 0;
    uint32_t shape_count = 1;

    for (size_t i = 0; i < sizes.size(); i++) {
        if (sizes[i] == -1) {
            if (need_recalc) {
                CLDNN_ERROR_MESSAGE(node.id(), "Only one dimension of the new shape can be -1");
            }
            need_recalc = i;
            continue;
        }
        if (sizes[i] == 0) {
            sizes[i] = input_sizes[i];
        }
        shape_count *= sizes[i];
    }
    if (need_recalc)
        sizes[need_recalc] = static_cast<int>(input_layout.size.count()) / shape_count;

    auto output_format = input_layout.format;
    auto output_dims_size = node.get_primitive()->output_shape_rank;

    if (output_dims_size == 6) {
        if (input_layout.get_rank() != 6) {
            output_format = cldnn::format::bfwzyx;
        }
    } else if (output_dims_size == 5) {
        if (input_layout.get_rank() != 5) {
            output_format = cldnn::format::bfzyx;
        }
    } else if (output_dims_size <= 4) {
        if (input_layout.get_rank() != 4) {
            output_format = cldnn::format::bfyx;
        }
    }

    return layout{input_layout.data_type, output_format, tensor(sizes)};
}

std::string reshape_inst::to_string(reshape_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite reshape_info;
    reshape_info.add("input id", input.id());
    reshape_info.add("output shape", desc->output_shape);

    node_info->add("reshape info", reshape_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reshape_inst::typed_primitive_inst(network& network, reshape_node const& node) : parent(network, node, false) {
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                    "Input layout data typr",
                                    input_layout.data_type,
                                    "output layout data type",
                                    output_layout.data_type,
                                    "");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Output layout count",
                          output_layout.count(),
                          "input layout count",
                          input_layout.count(),
                          "Output layout of reshape primitive changes size of input buffer");

    // if reshape operated in-place, postpone creation of the output until network run,
    // then create new memory object as the reinterpreted output of the previous primitive
    if (!node.can_be_optimized())
        _output = allocate_output();
    else
        reuse_input();
}

void reshape_inst::on_execute() {
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void reshape_inst::reuse_input() {
    build_deps();  // reshape need deps
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}

}  // namespace cldnn
