// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
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
    std::vector<tensor::value_type> dims_converted;
    for (auto dim : out_shape) {
        dims_converted.push_back(static_cast<tensor::value_type>(dim));
    }
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
    const auto& input0_layout = impl_param.get_input_layout(0);
    auto input0_shape = input0_layout.get<ShapeType>();

    auto& constant_mem = impl_param.memory_deps;
    const auto& begin_data = desc->begin;
    const auto& end_data = desc->end;
    const auto& strides_data = desc->strides;

    if ((begin_data.empty() && !constant_mem.count(1))
        || (end_data.empty() && !constant_mem.count(2))
        || (strides_data.empty() && !constant_mem.count(3))) {
        auto num_of_axis_mask_bit = [] (std::vector<int64_t> mask) -> size_t {
            size_t count = 0;
            for (size_t i = 0; i < mask.size(); i++)
                if (mask[i]) count++;
            return count;
        };

        const auto& input0_pshape = input0_layout.get_partial_shape();
        auto input0_len = input0_pshape.size();
        auto num_of_new_axis_bit = num_of_axis_mask_bit(desc->new_axis_mask);
        auto num_of_shrink_axis_bit = num_of_axis_mask_bit(desc->shrink_axis_mask);

        auto output_len = input0_len;
        if (num_of_new_axis_bit)
            output_len += num_of_new_axis_bit;
        else if (num_of_shrink_axis_bit)
            output_len -= num_of_shrink_axis_bit;

        auto output_shape = ov::PartialShape::dynamic(output_len);
        if (input0_layout.is_dynamic()) {
            // Fill with static shape until it finds dynamic while scaling output shape size based on the attributes
            //    1) new_axis_mask    : increase by the number of bits in new_axis_mask.
            //    2) shrink_axis_mask : decrease by the number of bits in shrink_mask.
            //
            // Notice :
            //     According to op implementation,
            //     new_axis_mask has higher priority than shrink_axis_mask,
            //     so that in case where bits of new_axis_mask is at the same position to that of shrink_axis_mask,
            //     the bits of shrink_axis_mask are ignored.
            //
            // To-Do :
            //    1) Consider the case where ellipsis_mask is enabled.
            //    2) Consider each combination of new_axis_mask, shrink_axis_mask and ellipsis_mask.
            size_t output_idx = 0;
            size_t input_idx = 0;
            for (size_t i = 0; i < output_len; i++) {
                if (num_of_new_axis_bit && desc->new_axis_mask[i]) {
                    output_shape[output_idx++] = {1};
                    continue;
                } else if (num_of_shrink_axis_bit && desc->shrink_axis_mask[i]) {
                    continue;
                }

                if (input0_pshape[input_idx].is_static()) {
                    output_shape[output_idx++] = input0_pshape[input_idx];
                    input_idx++;
                } else {
                    break;
                }
            }
        }

        return { layout{output_shape, input0_layout.data_type, format::get_default_format(output_len)} };
    }

    ov::op::v1::StridedSlice op;
    ShapeType begin_shape = begin_data.empty() ? impl_param.get_input_layout(1).get<ShapeType>() : ov::Shape{ begin_data.size() };
    ShapeType end_shape = end_data.empty() ? impl_param.get_input_layout(2).get<ShapeType>() : ov::Shape{ end_data.size() };
    ShapeType strides_shape = strides_data.empty() ? impl_param.get_input_layout(3).get<ShapeType>() : ov::Shape{ strides_data.size() };

    std::vector<ShapeType> output_shapes;
    std::vector<ShapeType> input_shapes = {
        std::move(input0_shape),
        begin_shape,
        end_shape,
        strides_shape
    };

    op.set_friendly_name(desc->id);
    op.set_begin_mask(desc->begin_mask);
    op.set_end_mask(desc->end_mask);
    op.set_new_axis_mask(desc->new_axis_mask);
    op.set_shrink_axis_mask(desc->shrink_axis_mask);
    op.set_ellipsis_mask_mask(desc->ellipsis_mask);

    std::unordered_map<size_t, ov::Tensor> const_data;
    const auto ta = ov::make_tensor_accessor(const_data);
    if (!begin_data.empty() && !end_data.empty() && !strides_data.empty()) {
        auto begin_tensor = make_tensor({ begin_shape, data_types::i64, format::bfyx }, const_cast<void*>(static_cast<const void*>(begin_data.data())));
        auto end_tensor = make_tensor({ end_shape, data_types::i64, format::bfyx }, const_cast<void*>(static_cast<const void*>(end_data.data())));
        auto strides_tensor = make_tensor({ strides_shape, data_types::i64, format::bfyx }, const_cast<void*>(static_cast<const void*>(strides_data.data())));

        const_data.emplace(1, begin_tensor);
        const_data.emplace(2, end_tensor);
        const_data.emplace(3, strides_tensor);

        output_shapes = ov::op::v1::shape_infer(&op, input_shapes, ta);
    } else {
        auto begin_mem = constant_mem.at(1);
        auto end_mem = constant_mem.at(2);
        auto strides_mem = constant_mem.at(3);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(begin_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(end_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(strides_mem, impl_param.get_stream());

        auto begin_tensor = make_tensor(begin_mem->get_layout(), lock1.data());
        auto end_tensor = make_tensor(end_mem->get_layout(), lock2.data());
        auto strides_tensor = make_tensor(strides_mem->get_layout(), lock3.data());

        const_data.emplace(1, begin_tensor);
        const_data.emplace(2, end_tensor);
        const_data.emplace(3, strides_tensor);

        output_shapes = ov::op::v1::shape_infer(&op, input_shapes, ta);
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
    std::vector<std::string> dependencies_info = {"begin_param id", "end_param id", "stride_param id"};
    for (size_t i = 1; i < node.get_dependencies().size(); ++i) {
        strided_slice_info.add(dependencies_info[i - 1], node.get_dependency(i).id());
    }
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

void strided_slice_inst::on_execute() {
    update_output_memory();
}

void strided_slice_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    if (node->get_program().is_new_shape_infer() && input_memory_ptr() == nullptr)
        return;

    if (static_cast<bool>(_outputs[0]) && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    GPU_DEBUG_TRACE_DETAIL << id() << " : update_output_memory with mem of input " << get_node().get_dependency(0).id()
                           << " : " << input_memory_ptr()->buffer_ptr() << std::endl;
    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }
    _outputs[0] = input_memory_ptr();
    _mem_allocated = false;
}

strided_slice_inst::typed_primitive_inst(network& network, strided_slice_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
