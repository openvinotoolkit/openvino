// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "resample_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include <string>
#include "json_object.h"

#include "interpolate_shape_inference.hpp"

namespace cldnn {
primitive_type_id resample::type_id() {
    static primitive_type_base<resample> instance;
    return &instance;
}

layout resample_inst::calc_output_layout(resample_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<resample>();
    auto input_layout = impl_param.get_input_layout();

    auto output_type = input_layout.data_type;
    if ((input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8)
        && desc->operation_type != resample::InterpolateOp::InterpolateMode::NEAREST) {
        output_type = data_types::f32;
    }
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    return desc->output_pattern.empty() ? layout({output_type, input_layout.format, desc->output_size}) :
                                          layout({desc->output_pattern, output_type, input_layout.format});
}

template<typename ShapeType>
std::vector<layout> resample_inst::calc_output_layouts(resample_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<resample>();
    auto input_layout = impl_param.get_input_layout(0);

    auto& memory_deps = impl_param.memory_deps;

    ov::op::v4::Interpolate op;
    op.set_attrs(desc->get_attrs());

    ShapeType pattern_shape = impl_param.input_layouts.size() == 2 ? impl_param.input_layouts[1].get<ShapeType>()
                                                                   : ov::Shape{ desc->output_pattern.size() };
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        impl_param.input_layouts[0].get<ShapeType>(),
        pattern_shape,
        ov::Shape{ desc->scales.size() },
        ov::Shape{ desc->axes.size() }
    };

    std::map<size_t, ngraph::HostTensorPtr> const_data;

    auto scales_data = desc->scales;
    auto scales_tensor = make_host_tensor({ ov::PartialShape{ ov::Shape{scales_data.size()} }, data_types::f32, format::bfyx },
                                            static_cast<void*>(scales_data.data()));
    const_data.emplace(2, scales_tensor);

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

    auto pattern_data = desc->output_pattern;
    if (memory_deps.count(1)) {
        auto pattern_mem = memory_deps.at(1);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> pattern_lock(pattern_mem, impl_param.prog.get_stream());

        auto pattern_ptr = pattern_lock.data();
        auto pattern_tensor = make_host_tensor(pattern_mem->get_layout(), pattern_ptr);

        const_data.emplace(1, pattern_tensor);
        ov::op::v4::shape_infer(&op, pads_begin, pads_end, input_shapes, output_shapes, {const_data});
    } else {
        auto pattern_tensor = make_host_tensor({ pattern_shape, data_types::i64, format::bfyx },
                                                 static_cast<void*>(pattern_data.data()));
        const_data.emplace(1, pattern_tensor);
        ov::op::v4::shape_infer(&op, pads_begin, pads_end, input_shapes, output_shapes, {const_data});
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
    resample_info.add("output padding lower size", desc->output_padding.lower_size());
    resample_info.add("output padding upper size", desc->output_padding.upper_size());

    node_info->add("resample_info", resample_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

resample_inst::typed_primitive_inst(network& network, resample_node const& node) : parent(network, node) {
}
}  // namespace cldnn
