// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

#include "strided_slice_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(strided_slice)

layout strided_slice_inst::calc_output_layout(strided_slice_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<strided_slice>();
    auto input_layout = impl_param.get_input_layout();
    auto output_format = format::get_default_format(desc->out_size.size());
    auto out_shape = desc->out_size;
    std::vector<tensor::value_type> dims_converted(out_shape.begin(), out_shape.end());
    // extend shape to 4d
    for (size_t i = dims_converted.size(); i < 4; i++) {
        dims_converted.push_back(1);
    }
    auto out_size = cldnn::tensor(output_format, dims_converted);
    return layout{input_layout.data_type, output_format, out_size};
}

template<typename ShapeType>
std::vector<layout> strided_slice_inst::calc_output_layouts(strided_slice_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<strided_slice>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input0_shape = input0_layout.get<ShapeType>();

    auto& constant_mem = impl_param.memory_deps;
    auto begin_data = desc->begin;
    auto end_data = desc->end;
    auto strides_data = desc->strides;

    if ((begin_data.empty() && !constant_mem.count(1))
        || (end_data.empty() && !constant_mem.count(2))
        || (strides_data.empty() && !constant_mem.count(3))) {
        auto out_shape = ov::PartialShape::dynamic(input0_layout.get_partial_shape().size());
        return { layout{out_shape, input0_layout.data_type, format::get_default_format(out_shape.rank().get_length())} };
    }

    ov::op::v1::StridedSlice op;
    ShapeType begin_shape = begin_data.empty() ? impl_param.get_input_layout(1).get<ShapeType>() : ov::Shape{ begin_data.size() };
    ShapeType end_shape = end_data.empty() ? impl_param.get_input_layout(2).get<ShapeType>() : ov::Shape{ end_data.size() };
    ShapeType strides_shape = strides_data.empty() ? impl_param.get_input_layout(3).get<ShapeType>() : ov::Shape{ strides_data.size() };

    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input0_shape,
        begin_shape,
        end_shape,
        strides_shape
    };

    op.set_begin_mask(desc->begin_mask);
    op.set_end_mask(desc->end_mask);
    op.set_new_axis_mask(desc->new_axis_mask);
    op.set_shrink_axis_mask(desc->shrink_axis_mask);
    op.set_ellipsis_mask_mask(desc->ellipsis_mask);

    std::map<size_t, ngraph::HostTensorPtr> const_data;
    if (!begin_data.empty() && !end_data.empty() && !strides_data.empty()) {
        auto begin_tensor = make_host_tensor({ begin_shape, data_types::i64, format::bfyx }, static_cast<void*>(begin_data.data()));
        auto end_tensor = make_host_tensor({ end_shape, data_types::i64, format::bfyx }, static_cast<void*>(end_data.data()));
        auto strides_tensor = make_host_tensor({ strides_shape, data_types::i64, format::bfyx }, static_cast<void*>(strides_data.data()));

        const_data.emplace(1, begin_tensor);
        const_data.emplace(2, end_tensor);
        const_data.emplace(3, strides_tensor);

        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
    } else {
        auto begin_mem = constant_mem.at(1);
        auto end_mem = constant_mem.at(2);
        auto strides_mem = constant_mem.at(3);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(begin_mem, impl_param.prog->get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(end_mem, impl_param.prog->get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(strides_mem, impl_param.prog->get_stream());

        auto begin_tensor = make_host_tensor(begin_mem->get_layout(), lock1.data());
        auto end_tensor = make_host_tensor(end_mem->get_layout(), lock2.data());
        auto strides_tensor = make_host_tensor(strides_mem->get_layout(), lock3.data());

        const_data.emplace(1, begin_tensor);
        const_data.emplace(2, end_tensor);
        const_data.emplace(3, strides_tensor);

        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
    }

    auto output_format = format::get_default_format(output_shapes[0].size());

    return { layout{output_shapes[0], input0_layout.data_type, output_format} };
}

template std::vector<layout> strided_slice_inst::calc_output_layouts<ov::PartialShape>(strided_slice_node const& node, const kernel_impl_params& impl_param);

std::string strided_slice_inst::to_string(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite strided_slice_info;
    strided_slice_info.add("input id", input.id());
    strided_slice_info.add("begin_param id", node.get_dependency(1).id());
    strided_slice_info.add("end_param id", node.get_dependency(2).id());
    strided_slice_info.add("stride_param id", node.get_dependency(3).id());
    strided_slice_info.add("begin", node.get_primitive()->begin);
    strided_slice_info.add("end", node.get_primitive()->end);
    strided_slice_info.add("strides", node.get_primitive()->strides);
    strided_slice_info.add("begin mask", node.get_primitive()->begin_mask);
    strided_slice_info.add("end mask", node.get_primitive()->end_mask);
    strided_slice_info.add("new axis mask", node.get_primitive()->new_axis_mask);
    strided_slice_info.add("shrink axis mask", node.get_primitive()->shrink_axis_mask);
    strided_slice_info.add("ellipsis mask", node.get_primitive()->ellipsis_mask);

    node_info->add("strided_slice info", strided_slice_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

strided_slice_inst::typed_primitive_inst(network& network, strided_slice_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
