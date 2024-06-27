// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_inst.h"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "pad_shape_inference.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(border)

layout border_inst::calc_output_layout(border_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for border_node!");
    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;
    auto desc = impl_param.typed_desc<border>();

    auto dims_format = format::adjust_to_rank(format::bfyx, input_layout.get_rank());
    auto new_dims = input_layout.get_dims();

    for (size_t i = 0; i < new_dims.size(); ++i) {
        new_dims[i] += (i < desc->pads_begin.size()) ? desc->pads_begin[i] : 0;
        new_dims[i] += (i < desc->pads_end.size()) ? desc->pads_end[i] : 0;
    }
    return layout{ input_layout.data_type, input_format, tensor(dims_format, new_dims) };
}

template<typename ShapeType>
std::vector<layout> border_inst::calc_output_layouts(border_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<border>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto output_type = input0_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    size_t in_rank = input0_layout.get_partial_shape().size();

    ov::op::v1::Pad op;
    op.set_pad_mode(desc->pad_mode);

    const bool is_begin_mem = (desc->non_constant_input_mask & border::PAD_NON_CONST_INPUT::BEGIN);
    const bool is_end_mem = (desc->non_constant_input_mask & border::PAD_NON_CONST_INPUT::END);

    const size_t begin_mem_idx = is_begin_mem ? 1 : 0;
    const size_t end_mem_idx = is_begin_mem ? 2 : 1;

    auto& memory_deps = impl_param.memory_deps;
    if ((is_begin_mem && memory_deps.count(begin_mem_idx) == 0) ||
        (is_end_mem && memory_deps.count(end_mem_idx) == 0)) {
        return {layout{ShapeType::dynamic(static_cast<int64_t>(in_rank)), input0_layout.data_type, input0_layout.format}};
    }

    int64_t begin_size = desc->pads_begin.size();
    int64_t end_size = desc->pads_end.size();

    layout pads_begin_layout = is_begin_mem ? impl_param.get_input_layout(begin_mem_idx) : layout({ begin_size }, data_types::i64, format::bfyx);
    layout pads_end_layout = is_end_mem ? impl_param.get_input_layout(end_mem_idx) : layout({ end_size }, data_types::i64, format::bfyx);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        pads_begin_layout.get<ShapeType>(),
        pads_end_layout.get<ShapeType>(),
    };

    TensorsContainer const_data(&impl_param.get_stream());

    auto pads_begin_data = desc->pads_begin;
    auto pads_end_data = desc->pads_end;

    if (is_begin_mem) {
        const_data.emplace(1, memory_deps.at(begin_mem_idx));
    } else {
        const_data.emplace(1, make_tensor(pads_begin_layout, static_cast<void*>(pads_begin_data.data())));
    }

    if (is_end_mem) {
        const_data.emplace(2, memory_deps.at(end_mem_idx));
    } else {
        const_data.emplace(2, make_tensor(pads_end_layout, static_cast<void*>(pads_end_data.data())));
    }

    auto ta = cldnn::make_tensor_accessor(const_data);
    std::vector<ShapeType> output_shapes = ov::op::shape_infer(&op, input_shapes, ta);

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> border_inst::calc_output_layouts<ov::PartialShape>(border_node const& node, const kernel_impl_params& impl_param);

std::string border_inst::to_string(border_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite border_info;
    border_info.add("pads_begin", desc->pads_begin);
    border_info.add("pads_end", desc->pads_end);
    border_info.add("pad mode", desc->pad_mode);
    border_info.add("pad value", std::to_string(desc->pad_value));
    border_info.add("negative_pad", std::to_string(desc->allow_negative_pad));

    node_info->add("border info", border_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

border_inst::typed_primitive_inst(network& network, border_node const& node) : parent(network, node) {
    auto input_layout = node.get_input_layout();
    if (input_layout.is_dynamic()) {
        return;
    }

    const auto& input_sizes = input_layout.get_dims();
    const auto pad_mode = argument->pad_mode;
    const bool allow_negative_pad = argument->allow_negative_pad;

    const auto check_negative_pad = [](std::ptrdiff_t pad) {
                                        return pad < 0;
                                    };

    if (!allow_negative_pad) {
        // Check if sizes of border are in proper range.
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin border sizes",
                         std::any_of(argument->pads_begin.begin(), argument->pads_begin.end(), check_negative_pad),
                         "Invalid border size: negative value");
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_end border sizes",
                         std::any_of(argument->pads_end.begin(), argument->pads_end.end(), check_negative_pad),
                         "Invalid border size: negative value");
    }

    if (pad_mode == ov::op::PadMode::SYMMETRIC) {
        bool valid_pads = true;

        for (size_t i = 0; i < argument->pads_begin.size(); ++i) {
            valid_pads &= argument->pads_begin[i] <= input_sizes[i];
            valid_pads &= argument->pads_end[i] <= input_sizes[i];
        }
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin/pads_end border sizes",
                         !valid_pads,
                         "Not enough data in input to create SYMMETRIC border of specified size");
    } else if (pad_mode == ov::op::PadMode::REFLECT) {
        bool valid_pads = true;

        for (size_t i = 0; i < argument->pads_begin.size(); ++i) {
            valid_pads &= argument->pads_begin[i] < input_sizes[i];
            valid_pads &= argument->pads_end[i] < input_sizes[i];
        }
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin/pads_end border sizes",
                         !valid_pads,
                         "Not enough data in input to create REFLECT border of specified size");
    }
}
}  // namespace cldnn
