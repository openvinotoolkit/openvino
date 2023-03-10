// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "resample_inst.h"
#include "primitive_type_base.h"
#include <string>
#include "json_object.h"

#include "interpolate_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(resample)

layout resample_inst::calc_output_layout(resample_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<resample>();
    auto input_layout = impl_param.get_input_layout();

    auto output_type = input_layout.data_type;
    if ((input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8)
        && desc->operation_type != resample::InterpolateOp::InterpolateMode::NEAREST
        && desc->operation_type != resample::InterpolateOp::InterpolateMode::LINEAR_ONNX) {
        output_type = data_types::f32;
    }
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    return desc->sizes.empty() ? layout({output_type, input_layout.format, desc->output_size}) :
                                 layout({desc->sizes, output_type, input_layout.format});
}

template<typename ShapeType>
std::vector<layout> resample_inst::calc_output_layouts(resample_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<resample>();
    auto input_layout = impl_param.get_input_layout(0);
    auto input_shape = input_layout.get<ShapeType>();
    size_t input_rank = input_shape.size();

    ov::op::v4::Interpolate op;
    op.set_attrs(desc->get_attrs());

    ShapeType sizes_shape = desc->sizes.empty() ? ov::Shape{ input_rank }
                                                : ov::Shape{ desc->sizes.size() };
    ShapeType scales_shape = desc->scales.empty() ? ov::Shape{ input_rank }
                                                  : ov::Shape{ desc->scales.size() };
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        input_shape,
        sizes_shape,
        scales_shape,
        ov::Shape{ desc->axes.size() }
    };

    auto& memory_deps = impl_param.memory_deps;
    std::map<size_t, ngraph::HostTensorPtr> const_data;

    auto sizes_data = desc->sizes;
    auto scales_data = desc->scales;

    bool sizes_calc_mod = desc->get_attrs().shape_calculation_mode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES;

    if (((sizes_data.empty() && !memory_deps.count(1)) || !sizes_calc_mod) &&
        ((scales_data.empty() && !memory_deps.count(2)) || sizes_calc_mod)) {
       return { layout{ShapeType::dynamic(input_rank), input_layout.data_type, input_layout.format} };
    }

    auto axes_data = desc->axes;
    if (axes_data.empty()) {
        axes_data.resize(input_layout.get_rank());
        std::iota(axes_data.begin(), axes_data.end(), 0);
    }
    auto axes_tensor = make_host_tensor({ ov::PartialShape{ ov::Shape{axes_data.size()} }, data_types::i64, format::bfyx },
                                          static_cast<void*>(axes_data.data()));
    const_data.emplace(3, axes_tensor);

    auto pads_begin = desc->pads_begin;
    auto pads_end = desc->pads_end;
    ov::op::v4::correct_pads_attr(&op, pads_begin, pads_end, input_shapes);

    if (sizes_calc_mod) {
        if (!sizes_data.empty()) {
            auto sizes_tensor = make_host_tensor({ sizes_shape, data_types::i64, format::bfyx }, static_cast<void*>(sizes_data.data()));
            const_data.emplace(1, sizes_tensor);
            ov::op::v4::shape_infer(&op, pads_begin, pads_end, input_shapes, output_shapes, {const_data});
        } else {
            auto sizes_mem = memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(sizes_mem, impl_param.prog->get_stream());
            auto sizes_tensor = make_host_tensor(sizes_mem->get_layout(), lock.data());
            const_data.emplace(1, sizes_tensor);
            ov::op::v4::shape_infer(&op, pads_begin, pads_end, input_shapes, output_shapes, {const_data});
        }
    } else {
        if (!scales_data.empty()) {
            auto scales_tensor = make_host_tensor({ scales_shape, data_types::f32, format::bfyx }, static_cast<void*>(scales_data.data()));
            const_data.emplace(2, scales_tensor);
            ov::op::v4::shape_infer(&op, pads_begin, pads_end, input_shapes, output_shapes, {const_data});
        } else {
            auto scales_mem = memory_deps.at(2);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(scales_mem, impl_param.prog->get_stream());
            auto scales_tensor = make_host_tensor(scales_mem->get_layout(), lock.data());
            const_data.emplace(2, scales_tensor);
            ov::op::v4::shape_infer(&op, pads_begin, pads_end, input_shapes, output_shapes, {const_data});
        }
    }

    return { layout{output_shapes[0], input_layout.data_type, format::adjust_to_rank(input_layout.format, output_shapes[0].size())} };
}

template std::vector<layout> resample_inst::calc_output_layouts<ov::PartialShape>(resample_node const& node, const kernel_impl_params& impl_param);

std::string resample_inst::to_string(resample_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite resample_info;
    if (desc->operation_type == resample::InterpolateOp::InterpolateMode::NEAREST)
        resample_info.add("resample_type:", "nearest_neighbor");
    else if (desc->operation_type == resample::InterpolateOp::InterpolateMode::LINEAR)
        resample_info.add("resample_type:", "caffe_bilinear_interp");
    else if (desc->operation_type == resample::InterpolateOp::InterpolateMode::CUBIC)
        resample_info.add("resample_type:", "cubic");
    else if (desc->operation_type == resample::InterpolateOp::InterpolateMode::LINEAR_ONNX)
        resample_info.add("resample_type:", "linear_onnx");
    else
        resample_info.add("resample_type:", "not supported sample type");

    if (desc->shape_calc_mode == resample::InterpolateOp::ShapeCalcMode::SIZES)
        resample_info.add("shape_calculation_mode:", "sizes");
    else
        resample_info.add("shape_calculation_mode:", "scales");

    if (desc->shape_calc_mode == resample::InterpolateOp::ShapeCalcMode::SCALES) {
        std::string axesAndScalesDump;
        std::string delim = "";
        for (size_t i = 0; i < desc->axes.size(); i++) {
            axesAndScalesDump += delim;
            delim = ", ";
            axesAndScalesDump += std::to_string(desc->axes[i]) + ": ";
            axesAndScalesDump += std::to_string(desc->scales[i]);
        }
        resample_info.add("scales:", axesAndScalesDump);
    }

    if (desc->coord_trans_mode == resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL)
        resample_info.add("coordinate_transformation_mode:", "half_pixel");
    else if (desc->coord_trans_mode == resample::InterpolateOp::CoordinateTransformMode::PYTORCH_HALF_PIXEL)
        resample_info.add("coordinate_transformation_mode:", "pytorch_half_pixel");
    else if (desc->coord_trans_mode == resample::InterpolateOp::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN)
        resample_info.add("coordinate_transformation_mode:", "tf_half_pixel_for_nn");
    else if (desc->coord_trans_mode == resample::InterpolateOp::CoordinateTransformMode::ALIGN_CORNERS)
        resample_info.add("coordinate_transformation_mode:", "align_corners");
    else
        resample_info.add("coordinate_transformation_mode:", "asymmetric");

    if (desc->round_mode == resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR)
        resample_info.add("nearest_mode:", "round_prefer_floor");
    if (desc->round_mode == resample::InterpolateOp::NearestMode::ROUND_PREFER_CEIL)
        resample_info.add("nearest_mode:", "round_prefer_ceil");
    if (desc->round_mode == resample::InterpolateOp::NearestMode::FLOOR)
        resample_info.add("nearest_mode:", "floor");
    if (desc->round_mode == resample::InterpolateOp::NearestMode::CEIL)
        resample_info.add("nearest_mode:", "ceil");
    else
        resample_info.add("nearest_mode:", "simple");

    resample_info.add("output_size", desc->output_size);
    resample_info.add("output padding lower size", desc->output_paddings[0].lower_size());
    resample_info.add("output padding upper size", desc->output_paddings[0].upper_size());

    node_info->add("resample_info", resample_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

resample_inst::typed_primitive_inst(network& network, resample_node const& node) : parent(network, node) {
}
}  // namespace cldnn
