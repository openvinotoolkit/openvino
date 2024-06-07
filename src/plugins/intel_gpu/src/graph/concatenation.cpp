// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_inst.h"
#include "concat_shape_inference.hpp"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>
#include <memory>
#include <list>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(concatenation)

layout concatenation_inst::calc_output_layout(concatenation_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<concatenation>();

    auto input_layout = impl_param.get_input_layout();
    auto output_format = input_layout.format;
    auto result_sizes = input_layout.get_dims();

    auto output_dt = desc->output_data_types[0].value_or(input_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_dt = impl_param.get_output_element_type();
    }

    auto axis_index = desc->axis;

    // calculate sum of features from all inputs
    result_sizes[axis_index] = 0;
    for (size_t i = 0; i < desc->input.size(); ++i) {
        auto input_sizes = impl_param.get_input_layout(i).get_dims();
        if (impl_param.get_input_layout(i).format == format::b_fs_yx_fsv16)
            output_format = format::b_fs_yx_fsv16;

        result_sizes[axis_index] += input_sizes[axis_index];
    }

    auto def_fmt = format::get_default_format(input_layout.get_rank());

    return layout {output_dt, output_format, tensor(def_fmt, result_sizes)};
}

template<typename ShapeType>
std::vector<layout> concatenation_inst::calc_output_layouts(const concatenation_node& /* node */, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<concatenation>();

    auto input_layout = impl_param.get_input_layout();

    auto output_dt = desc->output_data_types[0].value_or(input_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_dt = impl_param.get_output_element_type();
    }
    auto output_format = input_layout.format;
    for (size_t i = 0; i < desc->input.size(); ++i) {
        if (impl_param.get_input_layout(i).format == format::b_fs_yx_fsv16)
            output_format = format::b_fs_yx_fsv16;
    }

    auto axis_index = desc->axis;
    std::vector<ShapeType> input_shapes;

    for (size_t i = 0; i < desc->input.size(); ++i) {
        auto input_shape = impl_param.get_input_layout(i).get<ShapeType>();
        input_shapes.push_back(input_shape);
    }
    ov::op::v0::Concat op;
    op.set_friendly_name(desc->id);
    op.set_axis(axis_index);
    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);
    return { layout {output_shapes[0], output_dt, output_format} };
}

template std::vector<layout> concatenation_inst::calc_output_layouts<ov::PartialShape>(concatenation_node const& node, const kernel_impl_params& impl_param);

std::string concatenation_inst::to_string(concatenation_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream ss_inputs;
    std::stringstream primitive_description;

    for (size_t i = 0; i < node.get_inputs_count(); ++i) {
        ss_inputs << node.input(i).id();
        if (node.input(i).get_output_layout().is_static())
            ss_inputs << ", count: " << node.input(i).get_output_layout().count();
        else
            ss_inputs << ", count: " << "?";
        i != (node.get_inputs_count() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    json_composite concat_info;
    concat_info.add("concat axis", desc->axis);
    concat_info.add("inputs count", node.get_inputs_count());
    concat_info.add("inputs", ss_inputs.str());

    node_info->add("concat info", concat_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

concatenation_inst::typed_primitive_inst(network& network, concatenation_node const& node)
    : parent(network, node) {
    if (node.is_dynamic()) return;
    auto input_layout = node.get_input_layout();

    auto output_layout = node.get_output_layout();

    tensor::value_type concat_count = 0;
    auto input_size = input_layout.get_dims();
    auto output_size = output_layout.get_dims();
    for (const auto& i : node.get_dependencies()) {
        auto input_i_layout = i.first->get_output_layout();
        auto input_mem_size = input_i_layout.get_dims();
        for (int64_t dim = 0; dim < static_cast<int64_t>(output_layout.get_rank()); ++dim) {
            if (dim == node.get_primitive()->axis) {
                concat_count += input_mem_size[dim];
            } else {
                CLDNN_ERROR_NOT_EQUAL(node.id(),
                                      "Input size dim: " + std::to_string(dim),
                                      input_size[dim],
                                      "input memory dim: " + std::to_string(dim),
                                      input_mem_size[dim],
                                      "Every input must have the same size");
            }
        }
    }

    for (int64_t dim = 0; dim < static_cast<int64_t>(output_layout.get_rank()); ++dim) {
        if (dim == node.get_primitive()->axis) {
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Concat count",
                                  concat_count,
                                  "output size dim:" + std::to_string(dim),
                                  output_size[dim],
                                  "Output size in concatenated dimension mismatch sum of inputs!");
        } else {
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Input size dim: " + std::to_string(dim),
                                  input_size[dim],
                                  "output size dim:" + std::to_string(dim),
                                  output_size[dim],
                                  "Output size in non-concatenated dimension mistmatch input");
        }
    }

    if (node.can_be_optimized()) {
        build_deps();
        std::list<std::vector<std::pair<primitive_inst*, int32_t>>*> stack = {&_deps};
        while (!stack.empty()) {
            auto nodes_list = stack.front();
            stack.pop_front();

            for (const auto& processed_nodes : *nodes_list) {
                auto processed_node = processed_nodes.first;
                processed_node->_outputs = _outputs;
                if (processed_node->type() == concatenation::type_id() && processed_node->can_be_optimized()) {
                    if (!processed_node->_deps.empty())
                        stack.push_back(&processed_node->_deps);
                }
            }
        }
    }
}
}  // namespace cldnn
