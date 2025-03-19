// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "json_object.h"
#include <string>

#include "variadic_split_shape_inference.hpp"
#include "split_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(crop)

layout crop_inst::calc_output_layout(crop_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
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
    OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false,
           "Output data type forcing is not supported for crop_node!");

    auto desc = impl_param.typed_desc<crop>();
    const auto in_layout = impl_param.get_input_layout(0);
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        impl_param.input_layouts[0].get<ShapeType>(),
    };
    for (size_t i = 1; i < desc->input.size(); ++i) {
        input_shapes.push_back(impl_param.input_layouts[i].get<ShapeType>());
    }
    int64_t axis = desc->axis;

    // TODO: calling shape_infer for all cropped outpus is redundant... Need to optimize.
    if (desc->op_mode == cldnn::crop_ngraph_op_mode::variadic_split) {
        std::unordered_map<size_t, ov::Tensor> const_data;

        // If axis is negative value, it's not nomralized axis value, so it's non-constant input
        if (axis < 0) {
            OPENVINO_ASSERT(impl_param.memory_deps.count(1) > 0, "[GPU] Can't find Crop(ngraph VariadicSplit op mode) axis values memory dependency");
            auto axis_values_mem = impl_param.memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> axis_values_mem_lock(axis_values_mem, impl_param.get_stream());
            const_data.emplace(1, make_tensor(axis_values_mem->get_layout(), axis_values_mem_lock.data()));
        } else {
            const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, static_cast<void*>(&axis)));
        }
        if (impl_param.memory_deps.count(2) > 0) {
            auto split_length_mem = impl_param.memory_deps.at(2);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> split_length_mem_lock(split_length_mem, impl_param.get_stream());
            const_data.emplace(2, make_tensor(split_length_mem->get_layout(), split_length_mem_lock.data()));

            ov::op::v1::VariadicSplit op;
            op.set_friendly_name(desc->id);
            output_shapes = shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
        } else {
            auto input0_layout = impl_param.get_input_layout(0);
            auto out_shape = ov::PartialShape::dynamic(input0_layout.get_partial_shape().size());
            return { layout{out_shape, input0_layout.data_type, input0_layout.format } };
        }
    } else if (desc->op_mode == cldnn::crop_ngraph_op_mode::split) {
        std::unordered_map<size_t, ov::Tensor> const_data;

        // If axis is negative value, it's not nomralized axis value, so it's non-constant input
        if (axis < 0) {
            OPENVINO_ASSERT(impl_param.memory_deps.count(1) > 0, "[GPU] Can't find Crop(ngraph Split op mode) axis values memory dependency");
            auto axis_values_mem = impl_param.memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> axis_values_mem_lock(axis_values_mem, impl_param.get_stream());
            const_data.emplace(1, make_tensor(axis_values_mem->get_layout(), axis_values_mem_lock.data()));
        } else {
            const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, static_cast<void*>(&axis)));
        }

        ov::op::v1::Split op;
        op.set_friendly_name(desc->id);
        op.set_num_splits(desc->num_splits);
        output_shapes = shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    } else if (desc->op_mode == cldnn::crop_ngraph_op_mode::none) {
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

            return {layout{out_sizes.get_partial_shape(in_layout.get_partial_shape().size(), in_layout.get_rank()), in_layout.data_type, in_layout.format}};
        }
        return {layout{ref_in_sizes.get_partial_shape(in_layout.get_partial_shape().size(), in_layout.get_rank()), in_layout.data_type, in_layout.format}};
    }

    bool is_output_static = false;
    std::vector<layout> output_layouts;
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_layouts.push_back(layout({output_shapes[i], in_layout.data_type, in_layout.format}));
        is_output_static = (output_shapes[i].is_static()) ? true : is_output_static;
    }

    // update split offsets
    if (is_output_static) {
        auto p_param = const_cast<kernel_impl_params*>(&impl_param);
        ov::Shape startOffset(p_param->input_layouts[0].get_partial_shape().size());
        auto input_shape = p_param->input_layouts[0].get_partial_shape();
        auto dims = p_param->input_layouts[0].get_partial_shape().size();
        for (int32_t prev = 0; prev < desc->output_idx; prev++) {
            auto prev_crop_shape = output_layouts[prev].get_partial_shape().to_shape();
            for (size_t i = 0; i < dims; ++i) {
                if (prev_crop_shape[i] != input_shape.to_shape()[i])
                    startOffset[i] += prev_crop_shape[i];
            }
        }

        if (p_param->input_offsets.empty()) {
            p_param->input_offsets.resize(1);
            p_param->input_offsets[0] = desc->offsets;
        }

        p_param->input_offsets[0] = ov::intel_gpu::tensor_from_dims(startOffset, 0);
    }
    return {output_layouts[desc->output_idx]};
}

template std::vector<layout> crop_inst::calc_output_layouts<ov::PartialShape>(crop_node const& node, const kernel_impl_params& impl_param);

std::string crop_inst::to_string(crop_node const& node) {
    const auto& desc = node.get_primitive();
    auto ref_in_sizes = desc->reference_input;
    const auto& offsets = desc->offsets;
    const auto in_layout = node.get_input_layout();
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    json_composite crop_info;

    if (!in_layout.is_dynamic()) {
        const auto& in_sizes = in_layout.get_tensor();

        // Check for borders variant of crop.
        if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
            ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
            // Ignore not supported dimensions.
            const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
            const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});
            ref_in_sizes = in_sizes - (rb_sizes + lt_sizes);
        }
    }
    crop_info.add("reference input size", ref_in_sizes.to_string());
    crop_info.add("offset", offsets.to_string());
    crop_info.add("axis", desc->axis);
    crop_info.add("num_splits", desc->num_splits);
    crop_info.add("output_idx", desc->output_idx);

    node_info->add("crop info", crop_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

crop_inst::typed_primitive_inst(network& network, crop_node const& node) : parent(network, node) {
    const auto& ref_in_sizes = argument->reference_input;
    const auto in_layout = node.get_input_layout();
    const auto& offsets = argument->offsets;
    tensor null_tensor {};
    tensor value_tensor { 1, 1, 1, 1, 1 };

    if (in_layout.is_static()) {
        const auto& in_sizes = in_layout.get_tensor();
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
                                            "Reference input tensor/ input tensor mismatch");

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
    }

    if (!node.is_dynamic() && node.can_be_optimized()) {
        update_output_memory();
    }
}

void crop_inst::on_execute() {
    update_output_memory();
}

void crop_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    if (_node != nullptr)
        build_deps();

    if (node->get_program().is_new_shape_infer() && input_memory_ptr() == nullptr)
        return;

    if (_outputs[0] && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }
    _outputs[0] = _network.get_engine().reinterpret_buffer(input_memory(), _impl_params->get_output_layout());
    _mem_allocated = false;
}

}  // namespace cldnn
