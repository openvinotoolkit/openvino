// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
primitive_type_id border::type_id() {
    static primitive_type_base<border> instance;
    return &instance;
}

layout border_inst::calc_output_layout(border_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for border_node!");
    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;
    auto desc = impl_param.typed_desc<border>();

    auto dims_format = format::adjust_to_rank(format::bfyx, input_layout.get_rank());
    auto new_dims = input_layout.get_dims();

    for (size_t i = 0; i < new_dims.size(); ++i) {
        new_dims[i] += desc->pads_begin[i];
        new_dims[i] += desc->pads_end[i];
    }
    return layout{ input_layout.data_type, input_format, tensor(dims_format, new_dims) };
}

std::string border_inst::to_string(border_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite border_info;
    border_info.add("pads_begin", desc->pads_begin);
    border_info.add("pads_end", desc->pads_end);
    border_info.add("pad mode", desc->pad_mode);
    border_info.add("pad value", std::to_string(desc->pad_value));

    node_info->add("border info", border_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

border_inst::typed_primitive_inst(network& network, border_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();

    const auto& input_sizes = input_layout.get_dims();
    auto pad_mode = argument.pad_mode;

    // Check if sizes of border are in proper range.
    CLDNN_ERROR_BOOL(node.id(),
                     "pads_begin border sizes",
                     std::any_of(argument.pads_begin.begin(), argument.pads_begin.end(),
                                 [](std::ptrdiff_t pad) {
                                    return pad < 0;
                                }),
                     "Invalid border size: negative value");
    CLDNN_ERROR_BOOL(node.id(),
                     "pads_end border sizes",
                     std::any_of(argument.pads_end.begin(), argument.pads_end.end(),
                                 [](std::ptrdiff_t pad) {
                                    return pad < 0;
                                }),
                     "Invalid border size: negative value");

    if (pad_mode == ov::op::PadMode::SYMMETRIC) {
        bool valid_pads = true;

        for (size_t i = 0; i < input_sizes.size(); ++i) {
            valid_pads &= argument.pads_begin[i] <= input_sizes[i];
            valid_pads &= argument.pads_end[i] <= input_sizes[i];
        }
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin/pads_end border sizes",
                         !valid_pads,
                         "Not enough data in input to create SYMMETRIC border of specified size");
    } else if (pad_mode == ov::op::PadMode::REFLECT) {
        bool valid_pads = true;

        for (size_t i = 0; i < input_sizes.size(); ++i) {
            valid_pads &= argument.pads_begin[i] < input_sizes[i];
            valid_pads &= argument.pads_end[i] < input_sizes[i];
        }
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin/pads_end border sizes",
                         !valid_pads,
                         "Not enough data in input to create REFLECT border of specified size");
    }
}
}  // namespace cldnn
