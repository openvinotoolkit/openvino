// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_inst.h"
#include "pad_shape_inference.hpp"

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

template<typename ShapeType>
std::vector<layout> border_inst::calc_output_layouts(border_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<border>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto output_type = input0_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ov::op::v1::Pad op;
    op.set_pad_mode(desc->pad_mode);

    ShapeType pads_shape = impl_param.input_layouts.size() > 1 ? impl_param.get_input_layout(1).get<ShapeType>()
                                                               : ov::Shape{ desc->pads_begin.size() };
    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        pads_shape,
        pads_shape,
    };

    auto& memory_deps = impl_param.memory_deps;
    std::map<size_t, ngraph::HostTensorPtr> const_data;

    if (memory_deps.count(1) && memory_deps.count(2)) {
        auto pads_begin_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> pads_begin_lock(pads_begin_mem, impl_param.prog.get_stream());
        const_data.emplace(1, make_host_tensor(pads_begin_mem->get_layout(), pads_begin_lock.data()));

        auto pads_end_mem = memory_deps.at(2);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> pads_end_lock(pads_end_mem, impl_param.prog.get_stream());
        const_data.emplace(2, make_host_tensor(pads_end_mem->get_layout(), pads_end_lock.data()));

        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
    } else {
        auto pads_begin_data = desc->pads_begin;
        auto pads_begin_tensor = make_host_tensor({pads_shape, data_types::i64, format::bfyx}, static_cast<void*>(pads_begin_data.data()));
        const_data.emplace(1, pads_begin_tensor);

        auto pads_end_data = desc->pads_end;
        auto pads_end_tensor = make_host_tensor({pads_shape, data_types::i64, format::bfyx}, static_cast<void*>(pads_end_data.data()));
        const_data.emplace(2, pads_end_tensor);

        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
    }
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
