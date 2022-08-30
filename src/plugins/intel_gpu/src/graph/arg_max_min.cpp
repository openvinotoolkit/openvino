// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils_legacy.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <limits>

namespace cldnn {
primitive_type_id arg_max_min::type_id() {
    static primitive_type_base<arg_max_min> instance;
    return &instance;
}

layout arg_max_min_inst::calc_output_layout(arg_max_min_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<arg_max_min>();
    auto input_layout = impl_param.get_input_layout();
    bool values_first = desc->values_first;
    data_types output_data_type;
    data_types output_idx_type;
    output_data_type = desc->output_data_type ? *desc->output_data_type : input_layout.data_type;
    if (impl_param.input_layouts.size() == 3) {
        output_idx_type = impl_param.get_input_layout(2).data_type;
    } else {
        output_idx_type = *(desc->output_data_type);
    }
    auto size_check = [&](size_t tensor_size) {
        if (desc->input.size() == 1 && values_first)
            return;
        size_t max_size;
        // lowest integer not representable in floating point type = 2^(mantissa_bits + 1) + 1
        // https://stackoverflow.com/questions/3793838/which-is-the-first-integer-that-an-ieee-754-float-is-incapable-of-representing-e
        if (output_idx_type == data_types::f32) {
            max_size = (1 << std::numeric_limits<float>::digits);
        } else if (output_idx_type == data_types::f16) {
            // mantissa_bits for fp16 = 10
            max_size = (1 << 11);
        } else if (output_idx_type == data_types::u8) {
            max_size = std::numeric_limits<uint8_t>::max();
        } else if (output_idx_type == data_types::i32) {
            max_size = std::numeric_limits<int32_t>::max();
        } else {
            max_size = std::numeric_limits<size_t>::max();
        }

        if (tensor_size > max_size) {
            CLDNN_ERROR_GREATER_THAN(desc->id,
                                     "Reduced tensor size",
                                     tensor_size,
                                     "Maximum output data type value",
                                     max_size,
                                     "Current output data type is unable to hold maximum index of a tensor.");
        }
    };
    for (auto dim : input_layout.get_dims()) {
        size_check(dim);
    }
    auto format = input_layout.format;
    auto sizes = input_layout.get_dims();
    if (desc->axis >= static_cast<int64_t>(sizes.size()) || desc->axis < 0) {
        IE_THROW() << "Incorrect arg_max_min axis.";
    }
    sizes[desc->axis] = desc->top_k;
    return layout{output_data_type, format, tensor(format::get_default_format(input_layout.get_rank()), sizes)};
}

std::string arg_max_min_inst::to_string(arg_max_min_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("top_k", desc->top_k);
    conv_info.add("axis", desc->axis);
    conv_info.add("output type", desc->mode);
    conv_info.add("sort type", desc->sort);
    node_info->add("arg_max_min info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

arg_max_min_inst::typed_primitive_inst(network& network, arg_max_min_node const& node) : parent(network, node) {}
}  // namespace cldnn
