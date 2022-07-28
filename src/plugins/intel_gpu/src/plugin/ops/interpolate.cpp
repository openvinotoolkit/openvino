// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "caseless.hpp"

#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/resample.hpp"

namespace ov {
namespace intel_gpu {

static int64_t GetInterpolationAxis(int64_t axis, uint32_t rank) {
    if (axis < 0)
        axis += rank;

    if (axis < 0 || axis >= static_cast<int64_t>(rank))
        IE_THROW() << "Interpolate axis is not correspond to number of dimensions";

    return axis;
}

static void CreateInterpolateOp(Program& p, const std::shared_ptr<ngraph::op::v4::Interpolate>& op) {
    p.ValidateInputs(op, {3, 4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    static const size_t SCALES_INDEX = 2;
    static const size_t AXES_INDEX = 3;

    auto attrs = op->get_attrs();
    auto inputRank = op->get_input_shape(0).size();
    // auto outDims = op->get_output_shape(0).size();
    auto outTensor = tensor_from_dims(op->get_output_shape(0));

    auto scales_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(SCALES_INDEX));
    if (!scales_constant) {
        IE_THROW() << "Unsupported parameter node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    std::vector<float> scales = scales_constant->cast_vector<float>();

    std::vector<int64_t> axes;
    if (op->get_input_size() == 4) {
        auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(AXES_INDEX));
        if (!axes_constant) {
            IE_THROW() << "Unsupported parameter node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        auto ie_axes = axes_constant->cast_vector<int64_t>();
        for (auto axis : ie_axes) {
            axes.push_back(GetInterpolationAxis(axis, inputRank));
        }
    } else {
        for (size_t i = 0; i < inputRank; ++i) {
            axes.push_back(GetInterpolationAxis(i, inputRank));
        }
    }

    if (axes.size() != scales.size())
        IE_THROW() << op->get_friendly_name() << " Incorrect axes and scales should be the same size";

    auto interpolateMode = attrs.mode;
    if (interpolateMode == ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX) {
        if (inputRank != 2 && inputRank != 4)
            IE_THROW() << "mode 'linear_onnx' supports only 2D or 4D tensors";
        if (axes.size() != 2 && inputRank != axes.size())
            IE_THROW() << "mode 'linear_onnx' supports only axes with size 2 or equal to input rank";
        bool correctAxes =
            (axes[0] == 0 && axes[1] == 1) ||
            (axes[0] == 1 && axes[1] == 0) ||
            (axes[0] == 2 && axes[1] == 3) ||
            (axes[0] == 3 && axes[1] == 2);
        if (axes.size() == 4 && inputRank == 4) {
            for (size_t i = 0; i < axes.size(); i++) {
                if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                    correctAxes = false;
                    break;
                }
            }
        }
        if (!correctAxes)
            IE_THROW() <<
                "mode 'linear_onnx' supports only case when axes = {2, 3} or "
                "axes = {0, 1} or axes = {0, 1, 2, 3}";
    }

    auto resamplePrim = cldnn::resample(layerName,
                                        inputPrimitives[0],
                                        outTensor,
                                        scales,
                                        axes,
                                        attrs.pads_begin,
                                        attrs.pads_end,
                                        attrs.antialias,
                                        attrs.cube_coeff,
                                        interpolateMode,
                                        attrs.shape_calculation_mode,
                                        attrs.coordinate_transformation_mode,
                                        attrs.nearest_mode,
                                        op->get_friendly_name());

    p.AddPrimitive(resamplePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v4, Interpolate);

}  // namespace intel_gpu
}  // namespace ov
