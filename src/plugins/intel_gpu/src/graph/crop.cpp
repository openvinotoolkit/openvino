// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

#include "variadic_split_shape_inference.hpp"
#include "split_shape_inference.hpp"

namespace cldnn {
primitive_type_id crop::type_id() {
    static primitive_type_base<crop> instance;
    return &instance;
}

layout crop_inst::calc_output_layout(crop_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for crop_node!");
    auto desc = impl_param.typed_desc<crop>();
    const auto& ref_in_sizes = desc->reference_input;
    const auto in_layout = impl_param.get_input_layout();
    const auto& in_sizes = in_layout.get_tensor();
    const auto& offsets = desc->offsets;

    // Check for borders variant of crop.
    if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
        ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
        // Ignore not supported dimensions.
        const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
        const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});

        const auto out_sizes = in_sizes - (rb_sizes + lt_sizes);

        return layout({in_layout.data_type, in_layout.format, out_sizes});
    }
    return layout({in_layout.data_type, in_layout.format, ref_in_sizes});
}

template<typename ShapeType>
std::vector<layout> crop_inst::calc_output_layouts(const crop_node& /*node*/, const kernel_impl_params& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for crop_node!");

    auto desc = impl_param.typed_desc<crop>();
    const auto in_layout = impl_param.input_layouts[0];
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        impl_param.input_layouts[0].get<ShapeType>(),
    };
    for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
        input_shapes.push_back(impl_param.input_layouts[i].get<ShapeType>());
    }

    // TODO: calling shape_infer for all cropped outpus is redundant... Need to optimize.
    if (impl_param.input_layouts.size() == 3) {
        std::map<size_t, ngraph::HostTensorPtr> const_data;
        auto& memory_deps = impl_param.memory_deps;

        auto axis_values_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> axis_values_mem_lock(axis_values_mem, impl_param.prog.get_stream());
        const_data.emplace(1, make_host_tensor(axis_values_mem->get_layout(), axis_values_mem_lock.data()));

        auto split_length_mem = memory_deps.at(2);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> split_length_mem_lock(split_length_mem, impl_param.prog.get_stream());
        const_data.emplace(2, make_host_tensor(split_length_mem->get_layout(), split_length_mem_lock.data()));

        //VariadicSplit
        ov::op::v1::VariadicSplit op;
        shape_infer(&op, input_shapes, output_shapes, const_data);
    } else if (impl_param.input_layouts.size() == 2) {
        std::map<size_t, ngraph::HostTensorPtr> const_data;
        auto& memory_deps = impl_param.memory_deps;

        auto axis_values_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> axis_values_mem_lock(axis_values_mem, impl_param.prog.get_stream());
        const_data.emplace(1, make_host_tensor(axis_values_mem->get_layout(), axis_values_mem_lock.data()));

        // Split
        ov::op::v1::Split op;
        op.set_num_splits(impl_param.num_splits);
        shape_infer(&op, input_shapes, output_shapes, const_data);
    } else if (impl_param.input_layouts.size() == 1) {
        // Legacy usage
        if (in_layout.is_dynamic()) {
            auto in_shape = in_layout.get<ShapeType>();
            auto r = (in_shape.rank().is_static())? in_shape.size() : 1;
            return { layout{ShapeType::dynamic(r),
                    in_layout.data_type, in_layout.format.adjust_to_rank(in_layout.format, r)} };
        }

        const auto& ref_in_sizes = desc->reference_input;
        const auto& in_sizes = in_layout.get_tensor();
        const auto& offsets = desc->offsets;

        // Check for borders variant of crop.
        if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
            ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
            // Ignore not supported dimensions.
            const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
            const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});
            const auto out_sizes = in_sizes - (rb_sizes + lt_sizes);

            return {layout({in_layout.data_type, in_layout.format, out_sizes})};
        }
        return {layout({in_layout.data_type, in_layout.format, ref_in_sizes})};
    }
    return {layout({output_shapes[desc->output_idx], in_layout.data_type, in_layout.format})};
}

std::string crop_inst::to_string(crop_node const& node) {
    const auto& desc = node.get_primitive();
    auto ref_in_sizes = desc->reference_input;
    const auto& offsets = desc->offsets;
    const auto in_layout = node.input().get_output_layout();
    const auto& in_sizes = in_layout.get_tensor();

    auto node_info = node.desc_to_json();

    // Check for borders variant of crop.
    if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
        ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
        // Ignore not supported dimensions.
        const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
        const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});

        ref_in_sizes = in_sizes - (rb_sizes + lt_sizes);
    }

    std::stringstream primitive_description;

    json_composite crop_info;
    crop_info.add("reference input size", ref_in_sizes.to_string());
    crop_info.add("offset", offsets.to_string());

    node_info->add("crop info", crop_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

crop_inst::typed_primitive_inst(network& network, crop_node const& node) : parent(network, node) {
    const auto& ref_in_sizes = argument.reference_input;
    const auto in_layout = node.input().get_output_layout();
    const auto& in_sizes = in_layout.get_tensor();
    const auto& offsets = argument.offsets;
    tensor null_tensor {};
    tensor value_tensor { 1, 1, 1, 1, 1 };

    // Check for borders variant of crop.
    if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
        ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
        // Ignore not supported dimensions.
        const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
        const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});

        const auto out_sizes = in_sizes - (rb_sizes + lt_sizes);

        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Left/top/lower borders",
                                           lt_sizes,
                                           "0 value",
                                           null_tensor,
                                           "Invalid border size: negative");
        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Right/bottom/upper borders",
                                           rb_sizes,
                                           "0 value",
                                           null_tensor,
                                           "Invalid border size: negative");

        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Input sizes - border sizes",
                                           out_sizes,
                                           "1 value",
                                           value_tensor,
                                           "Invalid border sizes: greater-equal input sizes");
    }

    // check if output sizes matches reference input sizes
    CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                          "Reference input",
                                          ref_in_sizes,
                                          "input sizes",
                                          in_sizes,
                                          "Reference input tensor/ input tensor mismtach");

    // check if offsets do not extend input sizes and if match the output sizes
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                       "Batch offsets",
                                       offsets,
                                       "0 value",
                                       null_tensor,
                                       "Invalid Batch offset: negative value");
    auto input_size_sub_offsets = in_sizes - offsets;
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                       "input sizes - offsets",
                                       input_size_sub_offsets,
                                       "reference input sizes",
                                       ref_in_sizes,
                                       "Invalid Batch offset: exceeds data for output!");

    if (node.can_be_optimized()) {
        build_deps();
        reuse_input();
    }
}

void crop_inst::on_execute() {
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void crop_inst::reuse_input() {
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}
}  // namespace cldnn
