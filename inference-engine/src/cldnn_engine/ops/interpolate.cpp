// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "caseless.hpp"

#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/constant.hpp"

#include "api/resample.hpp"

namespace CLDNNPlugin {

static cldnn::coordinate_transformation_mode GetCoordinateTransformationMode(ngraph::op::v4::Interpolate::CoordinateTransformMode mode) {
    switch (mode) {
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel:
        return cldnn::coordinate_transformation_mode::half_pixel;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel:
        return cldnn::coordinate_transformation_mode::pytorch_half_pixel;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric:
        return cldnn::coordinate_transformation_mode::asymmetric;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn:
        return cldnn::coordinate_transformation_mode::tf_half_pixel_for_nn;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners:
        return cldnn::coordinate_transformation_mode::align_corners;
    }

    THROW_IE_EXCEPTION << "Unknown coordinate transformation mode: " << static_cast<int>(mode);
}

static cldnn::nearest_mode GetNearestMode(ngraph::op::v4::Interpolate::NearestMode mode) {
    switch (mode) {
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor:
        return cldnn::nearest_mode::round_prefer_floor;
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil:
        return cldnn::nearest_mode::round_prefer_ceil;
    case ngraph::op::v4::Interpolate::NearestMode::floor:
        return cldnn::nearest_mode::floor;
    case ngraph::op::v4::Interpolate::NearestMode::ceil:
        return cldnn::nearest_mode::ceil;
    case ngraph::op::v4::Interpolate::NearestMode::simple:
        return cldnn::nearest_mode::simple;
    }

    THROW_IE_EXCEPTION << "Unknown nearest mode: " << static_cast<int>(mode);
}

static cldnn::shape_calculation_mode GetShapeCalculationMode(ngraph::op::v4::Interpolate::ShapeCalcMode mode) {
    switch (mode) {
    case ngraph::op::v4::Interpolate::ShapeCalcMode::sizes:  return cldnn::shape_calculation_mode::sizes;
    case ngraph::op::v4::Interpolate::ShapeCalcMode::scales: return cldnn::shape_calculation_mode::scales;
    }
    THROW_IE_EXCEPTION << "Unknown shape calculation mode: " << static_cast<int>(mode);
}

static cldnn::resample_type GetResampleType(ngraph::op::v4::Interpolate::InterpolateMode mode) {
    switch (mode) {
    case ngraph::op::v4::Interpolate::InterpolateMode::nearest: return cldnn::resample_type::nearest;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear: return cldnn::resample_type::caffe_bilinear;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx: return cldnn::resample_type::linear_onnx;
    case ngraph::op::v4::Interpolate::InterpolateMode::cubic: return cldnn::resample_type::cubic;
    }
    THROW_IE_EXCEPTION << "Unknown interpolation mode: " << static_cast<int>(mode);
}

static cldnn::resample::resample_axis GetInterpolationAxis(int32_t axis, uint32_t sz) {
    if (axis < 0)
        axis += sz;
    if (axis < 0 || axis >= sz)
        THROW_IE_EXCEPTION << "Interpolate axis is not correspond to number of dimensions";

    // Difference in dimension ordering between IE and clDNN,
    // reverse spatial dimensions after batch and feature.
    uint32_t cldnn_axis = axis;
    if (axis >= 2) {
        auto spatial_axis = axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max(sz, 4u) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0:
            return cldnn::resample::resample_axis::along_b;
        case 1:
            return cldnn::resample::resample_axis::along_f;
        case 2:
            return cldnn::resample::resample_axis::along_x;
        case 3:
            return cldnn::resample::resample_axis::along_y;
        case 4:
            return cldnn::resample::resample_axis::along_z;
        case 5:
            return cldnn::resample::resample_axis::along_w;
        default:
            break;
    }
    THROW_IE_EXCEPTION << "Unsupported Interpolate axis: " << axis;
}

void CreateInterpolateOp(Program& p, const std::shared_ptr<ngraph::op::v4::Interpolate>& op) {
    p.ValidateInputs(op, {3, 4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    static const size_t SCALES_INDEX = 2;
    static const size_t AXES_INDEX = 3;

    auto attrs = op->get_attrs();
    auto inputRank = op->get_input_shape(0).size();
    auto outDims = op->get_output_shape(0).size();
    auto outTensor = CldnnTensorFromIEDims(op->get_output_shape(0));

    std::vector<int> pad_begin(attrs.pads_begin.begin(), attrs.pads_begin.end());
    std::vector<int> pad_end(attrs.pads_end.begin(), attrs.pads_end.end());

    for (size_t i = pad_begin.size(); i < outDims || i < 4; ++i)
        pad_begin.push_back(0);
    for (size_t i = pad_end.size(); i < outDims || i < 4; ++i)
        pad_end.push_back(0);

    int antialias = attrs.antialias;
    float cube_coeff = attrs.cube_coeff;

    auto cldnnSampleType = GetResampleType(attrs.mode);
    auto shapeCalcMode = GetShapeCalculationMode(attrs.shape_calculation_mode);
    auto coordTransMode = GetCoordinateTransformationMode(attrs.coordinate_transformation_mode);
    auto nearestMode = GetNearestMode(attrs.nearest_mode);

    auto scales_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(SCALES_INDEX));
    if (!scales_constant) {
        THROW_IE_EXCEPTION << "Unsupported parameter node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    std::vector<float> scales = scales_constant->cast_vector<float>();

    std::vector<cldnn::resample::resample_axis> axes;
    if (op->get_input_size() == 4) {
        auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(AXES_INDEX));
        if (!axes_constant) {
            THROW_IE_EXCEPTION << "Unsupported parameter node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        auto ie_axes = axes_constant->cast_vector<int32_t>();
        for (auto axis : ie_axes) {
            axes.push_back(GetInterpolationAxis(axis, inputRank));
        }
    } else {
        for (int i = 0; i < inputRank; ++i) {
            axes.push_back(GetInterpolationAxis(i, inputRank));
        }
    }

    if (axes.size() != scales.size())
        THROW_IE_EXCEPTION << op->get_friendly_name() << " Incorrect axes and scales should be the same size";

    cldnn::resample::AxesAndScales axesAndScales;
    for (size_t i = 0; i < axes.size(); ++i) {
        axesAndScales[axes[i]] = scales[i];
    }

    if (cldnnSampleType == cldnn::resample_type::linear_onnx) {
        if (inputRank != 2 && inputRank != 4)
            THROW_IE_EXCEPTION << "mode 'linear_onnx' supports only 2D or 4D tensors";
        if (axes.size() != 2 && inputRank != axes.size())
            THROW_IE_EXCEPTION << "mode 'linear_onnx' supports only axes with size 2 or equal to input rank";
        bool correctAxes =
            ((axes[0] == cldnn::resample::resample_axis::along_b) &&
             (axes[1] == cldnn::resample::resample_axis::along_f)) ||
            ((axes[0] == cldnn::resample::resample_axis::along_y) &&
             (axes[1] == cldnn::resample::resample_axis::along_x));
        if (axes.size() == 4 && inputRank == 4) {
            correctAxes = axes[0] == cldnn::resample::resample_axis::along_b &&
                          axes[1] == cldnn::resample::resample_axis::along_f &&
                          axes[2] == cldnn::resample::resample_axis::along_y &&
                          axes[3] == cldnn::resample::resample_axis::along_x;
        }
        if (!correctAxes)
            THROW_IE_EXCEPTION <<
                "mode 'linear_onnx' supports only case when axes = {2, 3} or "
                "axes = {0, 1} or axes = {0, 1, 2, 3}";
    }

    auto resamplePrim = cldnn::resample(layerName,
                                        inputPrimitives[0],
                                        outTensor,
                                        axesAndScales,
                                        pad_begin,
                                        pad_end,
                                        antialias,
                                        cube_coeff,
                                        cldnnSampleType,
                                        shapeCalcMode,
                                        coordTransMode,
                                        nearestMode);

    p.AddPrimitive(resamplePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v4, Interpolate);

}  // namespace CLDNNPlugin
