// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"

namespace ov::intel_gpu {

static std::vector<int64_t> ExtractAxes(const std::shared_ptr<ov::op::util::InterpolateBase>& op, size_t axes_index) {
    std::vector<int64_t> axes;
    auto inputRank = op->get_input_partial_shape(0).size();
    if (op->get_input_size() == axes_index + 1) {
        auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(axes_index));
        OPENVINO_ASSERT(axes_constant, "Unsupported parameter node type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        axes = axes_constant->cast_vector<int64_t>();
        ov::util::try_normalize_axes(axes, inputRank, *op);
    } else {
        for (size_t i = 0; i < inputRank; ++i) {
            axes.push_back(ov::util::try_normalize_axis(i, inputRank, *op));
        }
    }
    return axes;
}

static void ValidateAxesAndThrowIfError(const std::shared_ptr<ov::op::util::InterpolateBase>& op, const std::vector<int64_t>& axes) {
    auto inputRank = op->get_input_partial_shape(0).size();
    auto interpolateMode = op->get_attrs().mode;
    if (interpolateMode == ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX) {
        OPENVINO_ASSERT(inputRank == 2 || inputRank == 4 || inputRank == 5, "Mode 'linear_onnx' supports only 2D or 4D, 5D tensors");

        OPENVINO_ASSERT(axes.size() == 2 || axes.size() == 3 || inputRank == axes.size(),
                        "Mode 'linear_onnx' supports only axes with size 2, 3 or equal to input rank");

        bool correctAxes =
            ((axes.size() == 2 || axes.size() == 4) && inputRank < 5) &&
            ((axes[0] == 0 && axes[1] == 1) ||
             (axes[0] == 1 && axes[1] == 0) ||
             (axes[0] == 2 && axes[1] == 3) ||
             (axes[0] == 3 && axes[1] == 2));

        correctAxes |=
            (axes.size() == 3 || axes.size() == 5) && inputRank == 5 &&
            ((axes[0] == 0 && axes[1] == 1 && axes[2] == 2) ||
             (axes[0] == 2 && axes[1] == 3 && axes[2] == 4));

        if ((axes.size() == 4 && inputRank == 4) || (axes.size() == 5 && inputRank == 5)) {
            for (size_t i = 0; i < axes.size(); ++i) {
                if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                    correctAxes = false;
                    break;
                }
            }
        }
        OPENVINO_ASSERT(correctAxes, "Mode 'linear_onnx' supports only case when axes = {2, 3} or ",
                                     "axes = {0, 1} or axes = {0, 1, 2, 3} or axes = {2, 3, 4} for 5d");
    }
}

static void CreateInterpolateOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::Interpolate>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    static const size_t SIZES_INDEX = 1;
    static const size_t SCALES_INDEX = 2;
    static const size_t AXES_INDEX = 3;

    auto attrs = op->get_attrs();

    auto sizes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(SIZES_INDEX));
    std::vector<int64_t> sizes = sizes_constant ? sizes_constant->cast_vector<int64_t>() : std::vector<int64_t>{};

    auto scales_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(SCALES_INDEX));
    std::vector<float> scales = scales_constant ? scales_constant->cast_vector<float>() : std::vector<float>{};

    std::vector<int64_t> axes = ExtractAxes(op, AXES_INDEX);

    if (attrs.shape_calculation_mode == ov::op::v4::Interpolate::ShapeCalcMode::SCALES && scales_constant) {
        OPENVINO_ASSERT(axes.size() == scales.size(), "[GPU] Incorrect axes and scales values for Interpolate operation with id ", op->get_friendly_name());
    }

    // TODO shouldn't be all this checking done in ov::op::v4::Interpolate?
    ValidateAxesAndThrowIfError(op, axes);

    std::shared_ptr<cldnn::resample> resamplePrim = nullptr;
    if (p.use_new_shape_infer()) {
        if (sizes_constant && scales_constant) {
            resamplePrim = std::make_shared<cldnn::resample>(layerName,
                                                             inputs[0],
                                                             sizes,
                                                             scales,
                                                             axes,
                                                             attrs.pads_begin,
                                                             attrs.pads_end,
                                                             attrs.antialias,
                                                             attrs.cube_coeff,
                                                             attrs.mode,
                                                             attrs.shape_calculation_mode,
                                                             attrs.coordinate_transformation_mode,
                                                             attrs.nearest_mode);
        } else {
            resamplePrim = std::make_shared<cldnn::resample>(layerName,
                                                             inputs[0],
                                                             inputs[SIZES_INDEX],
                                                             inputs[SCALES_INDEX],
                                                             axes,
                                                             attrs.pads_begin,
                                                             attrs.pads_end,
                                                             attrs.antialias,
                                                             attrs.cube_coeff,
                                                             attrs.mode,
                                                             attrs.shape_calculation_mode,
                                                             attrs.coordinate_transformation_mode,
                                                             attrs.nearest_mode);
        }
    } else {
        auto outShape = op->get_output_shape(0);
        auto outputPattern = std::vector<int64_t>(outShape.begin(), outShape.end());

        resamplePrim = std::make_shared<cldnn::resample>(layerName,
                                                         inputs[0],
                                                         outputPattern,
                                                         scales,
                                                         axes,
                                                         attrs.pads_begin,
                                                         attrs.pads_end,
                                                         attrs.antialias,
                                                         attrs.cube_coeff,
                                                         attrs.mode,
                                                         attrs.shape_calculation_mode,
                                                         attrs.coordinate_transformation_mode,
                                                         attrs.nearest_mode);
    }
    p.add_primitive(*op, resamplePrim);
}

static void CreateInterpolateOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v11::Interpolate>& op) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    enum {
        eInputIndex,
        eScalesOrSizesIndex,
        eAxesIndex,
    };

    auto attrs = op->get_attrs();

    auto scales_or_sizes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(eScalesOrSizesIndex));
    std::vector<float> scales = scales_or_sizes_constant && attrs.shape_calculation_mode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES ?
        scales_or_sizes_constant->cast_vector<float>() : std::vector<float>{};
    std::vector<int64_t> sizes = scales_or_sizes_constant && attrs.shape_calculation_mode == ov::op::v11::Interpolate::ShapeCalcMode::SIZES ?
        scales_or_sizes_constant->cast_vector<int64_t>() : std::vector<int64_t>{};

    std::vector<int64_t> axes = ExtractAxes(op, eAxesIndex);

    if (attrs.shape_calculation_mode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES && scales_or_sizes_constant) {
        OPENVINO_ASSERT(axes.size() == scales.size(), "[GPU] Incorrect axes and scales values for Interpolate operation with id ", op->get_friendly_name());
    }

    // TODO shouldn't be all this checking done in ov::op::v4::Interpolate?
    ValidateAxesAndThrowIfError(op, axes);

    std::shared_ptr<cldnn::resample> resamplePrim = nullptr;
    if (p.use_new_shape_infer()) {
        if (scales_or_sizes_constant) {
            resamplePrim = std::make_shared<cldnn::resample>(layerName,
                                                             inputs[0],
                                                             sizes,
                                                             scales,
                                                             axes,
                                                             attrs.pads_begin,
                                                             attrs.pads_end,
                                                             attrs.antialias,
                                                             attrs.cube_coeff,
                                                             attrs.mode,
                                                             attrs.shape_calculation_mode,
                                                             attrs.coordinate_transformation_mode,
                                                             attrs.nearest_mode);
        } else {
            resamplePrim = std::make_shared<cldnn::resample>(layerName,
                                                             inputs[0],
                                                             inputs[eScalesOrSizesIndex],
                                                             inputs[eAxesIndex],
                                                             axes,
                                                             attrs.pads_begin,
                                                             attrs.pads_end,
                                                             attrs.antialias,
                                                             attrs.cube_coeff,
                                                             attrs.mode,
                                                             attrs.shape_calculation_mode,
                                                             attrs.coordinate_transformation_mode,
                                                             attrs.nearest_mode,
                                                             1);
        }
    } else {
        auto outShape = op->get_output_shape(0);
        auto outputPattern = std::vector<int64_t>(outShape.begin(), outShape.end());

        resamplePrim = std::make_shared<cldnn::resample>(layerName,
                                                         inputs[0],
                                                         outputPattern,
                                                         scales,
                                                         axes,
                                                         attrs.pads_begin,
                                                         attrs.pads_end,
                                                         attrs.antialias,
                                                         attrs.cube_coeff,
                                                         attrs.mode,
                                                         attrs.shape_calculation_mode,
                                                         attrs.coordinate_transformation_mode,
                                                         attrs.nearest_mode);
    }
    p.add_primitive(*op, resamplePrim);
}

REGISTER_FACTORY_IMPL(v4, Interpolate);
REGISTER_FACTORY_IMPL(v11, Interpolate);

}  // namespace ov::intel_gpu
