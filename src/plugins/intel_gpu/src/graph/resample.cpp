// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>

#include "interpolate_shape_inference.hpp"
#include "json_object.h"
#include "memory_accessor.hpp"
#include "primitive_type_base.h"
#include "resample_inst.h"

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
        output_type = impl_param.get_output_element_type();
    }

    return desc->sizes.empty() ? layout({output_type, input_layout.format, desc->output_size}) :
                                 layout({desc->sizes, output_type, input_layout.format});
}

namespace v4 {
template<typename ShapeType>
static std::vector<layout> calc_output_layouts(resample_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<resample>();
    auto input_layout = impl_param.get_input_layout(0);
    auto input_shape = input_layout.get<ShapeType>();
    size_t input_rank = input_shape.size();

    ov::op::v4::Interpolate op;
    op.set_attrs(desc->get_attrs());

    ShapeType sizes_shape = desc->sizes.empty() ? ov::Shape{ input_rank }
                                                : ov::Shape{ desc->sizes.size() };
    ShapeType scales_shape = desc->scales.empty() ? ov::Shape{input_rank} : ov::Shape{desc->scales.size()};
    std::vector<ShapeType> input_shapes = {input_shape, sizes_shape, scales_shape};

    std::unordered_map<size_t, ov::Tensor> tensors;

    auto sizes = desc->sizes;
    if (!sizes.empty()) {
        tensors.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{sizes.size()}, sizes.data()));
    }

    auto scales = desc->scales;
    if (!scales.empty()) {
        tensors.emplace(2, ov::Tensor(ov::element::f32, ov::Shape{scales.size()}, scales.data()));
    }

    auto axes = desc->axes;
    if (!axes.empty()) {
        auto axes_shape = ov::Shape{axes.size()};
        input_shapes.push_back(axes_shape);
        tensors.emplace(3, ov::Tensor(ov::element::i64, axes_shape, axes.data()));
    }

    auto& memory_deps = impl_param.memory_deps;
    const auto ta = MemoryAccessor(&memory_deps, impl_param.get_stream(), ov::make_tensor_accessor(tensors));

    auto pads_begin = desc->pads_begin;
    auto pads_end = desc->pads_end;
    const auto output_shapes = ov::op::v4::shape_infer(&op, input_shapes, pads_begin, pads_end, ta);

    auto output_type = input_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    return { layout{output_shapes[0], output_type, format::adjust_to_rank(input_layout.format, output_shapes[0].size())} };
}
} // namespace v4

namespace v11 {
template<typename ShapeType>
static std::vector<layout> calc_output_layouts(resample_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<resample>();
    auto input_layout = impl_param.get_input_layout(0);
    auto input_shape = input_layout.get<ShapeType>();
    size_t input_rank = input_shape.size();

    ov::op::v11::Interpolate op;
    op.set_attrs(desc->get_attrs());

    ShapeType sizes_or_scales_shape;
    if (!desc->sizes.empty()) {
        sizes_or_scales_shape = ov::Shape{ desc->sizes.size() };
    } else if (!desc->scales.empty()) {
        sizes_or_scales_shape = ov::Shape{ desc->scales.size() };
    } else {
        sizes_or_scales_shape = ov::Shape{ input_rank };
    }
    std::vector<ShapeType> input_shapes = {input_shape, sizes_or_scales_shape};

    std::unordered_map<size_t, ov::Tensor> tensors;

    auto sizes = desc->sizes;
    auto scales = desc->scales;
    if (!sizes.empty()) {
        tensors.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{sizes.size()}, sizes.data()));
    } else if (!scales.empty()) {
        tensors.emplace(1, ov::Tensor(ov::element::f32, ov::Shape{scales.size()}, scales.data()));
    }
    auto axes = desc->axes;
    if (!axes.empty()) {
        auto axes_shape = ov::Shape{axes.size()};
        input_shapes.push_back(axes_shape);
        tensors.emplace(2, ov::Tensor(ov::element::i64, axes_shape, axes.data()));
    }

    auto& memory_deps = impl_param.memory_deps;
    const auto ta = MemoryAccessor(&memory_deps, impl_param.get_stream(), ov::make_tensor_accessor(tensors));

    auto pads_begin = desc->pads_begin;
    auto pads_end = desc->pads_end;
    const auto output_shapes = ov::op::v11::shape_infer(&op, input_shapes, pads_begin, pads_end, ta);
    auto output_type = input_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }
    return { layout{output_shapes[0], output_type, format::adjust_to_rank(input_layout.format, output_shapes[0].size())} };
}
} // namespace v11

template<typename ShapeType>
std::vector<layout> resample_inst::calc_output_layouts(resample_node const& node, const kernel_impl_params& impl_param) {
    using Mode = ov::op::util::InterpolateBase::InterpolateMode;
    auto desc = impl_param.typed_desc<resample>();
    if (desc->operation_type == Mode::BILINEAR_PILLOW || desc->operation_type == Mode::BICUBIC_PILLOW)
        return v11::calc_output_layouts<ShapeType>(node, impl_param);
    else
        return v4::calc_output_layouts<ShapeType>(node, impl_param);
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
            if (desc->scales.size() > i)
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
    resample_info.add("output padding lower size", std::vector<tensor::value_type>(desc->output_paddings[0]._lower_size.begin(),
                                                                                   desc->output_paddings[0]._lower_size.end()));
    resample_info.add("output padding upper size", std::vector<tensor::value_type>(desc->output_paddings[0]._upper_size.begin(),
                                                                                   desc->output_paddings[0]._upper_size.end()));

    node_info->add("resample_info", resample_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

resample_inst::typed_primitive_inst(network& network, resample_node const& node) : parent(network, node) {
}
}  // namespace cldnn
