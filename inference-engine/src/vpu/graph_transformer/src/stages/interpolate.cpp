// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/stages/interpolate_stages.hpp>

#include <ie_common.h>
#include <ie_blob.h>
#include <ngraph/opsets/opset4.hpp>
#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

using namespace InferenceEngine;

namespace vpu {
static InterpolateCoordTransMode getVpuCoordTransMode (ngraph::op::v4::Interpolate::CoordinateTransformMode mode) {
    switch (mode)
    {
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners :
        return InterpolateCoordTransMode::AlignCorners;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric :
        return InterpolateCoordTransMode::Asymmetric;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel :
        return InterpolateCoordTransMode::HalfPixel;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel :
        return InterpolateCoordTransMode::PytorchHalfPixel;
    case ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn :
        return InterpolateCoordTransMode::TfHalfPixelForNn;
    default:
        VPU_THROW_UNLESS(false, "Can't convert ngraph CoordinateTransformMode to vpu mode");
        // should never be returned
        return InterpolateCoordTransMode::TfHalfPixelForNn;
    }
}
static InterpolateNearestMode getVpuNearestMode (ngraph::op::v4::Interpolate::NearestMode mode) {
    switch (mode)
    {
    case ngraph::op::v4::Interpolate::NearestMode::ceil :
        return InterpolateNearestMode::Ceil;
    case ngraph::op::v4::Interpolate::NearestMode::floor :
        return InterpolateNearestMode::Floor;
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor :
        return InterpolateNearestMode::RoundPreferFloor;
    case ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil :
        return InterpolateNearestMode::RoundPreferCeil;
    case ngraph::op::v4::Interpolate::NearestMode::simple :
        return InterpolateNearestMode::Simple;
    default:
        VPU_THROW_UNLESS(false, "Can't convert ngraph CoordinateTransformMode to vpu mode");
        // should never be returned
        return InterpolateNearestMode::Simple;
    }
}
static InterpolateMode getVpuInterpolateMode (ngraph::op::v4::Interpolate::InterpolateMode mode) {
    switch (mode)
    {
    case ngraph::op::v4::Interpolate::InterpolateMode::cubic :
        return InterpolateMode::Cubic;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear :
        return InterpolateMode::Linear;
    case ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx :
        return InterpolateMode::LinearOnnx;
    case ngraph::op::v4::Interpolate::InterpolateMode::nearest :
        return InterpolateMode::Nearest;

    default:
        VPU_THROW_UNLESS(false, "Can't convert ngraph CoordinateTransformMode to vpu mode");
        // should never be returned
        return InterpolateMode::Nearest;
    }
}
void FrontEnd::parseInterpolate(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& interpolate = ngraph::as_type_ptr<ngraph::op::v4::Interpolate>(node);
    IE_ASSERT(interpolate != nullptr);
    VPU_THROW_UNLESS(inputs.size() <= 4 && inputs.size() >= 1,
                     "Interpolate stage with name {} must have no more than 4 inputs and no less than 1 input, actually provided {} inputs",
                     interpolate->get_name(), inputs.size());

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Interpolate stage with name {} must have only 1 output, actually provided {} outputs",
                     interpolate->get_name(), outputs.size());
    const auto& attrs = interpolate->get_attrs();
    const auto interpolateMode = attrs.mode;

    const auto input  = inputs[0];
    const auto output = outputs[0];

    // try to use existing resize layers

    const auto oc = output->desc().dim(Dim::C);
    const auto on = output->desc().dim(Dim::N, 1);

    const auto ic = input->desc().dim(Dim::C);
    const auto in = input->desc().dim(Dim::N, 1);

    VPU_THROW_UNLESS(in == on, "incompatible: input batch=%d, output batch=%d", in, on);

    auto padsBegin = attrs.pads_begin;
    auto padsEnd   = attrs.pads_end;

    const auto isPadZeros = [](const std::vector<size_t>& pad) {
        return std::all_of(pad.begin(), pad.end(), [](size_t i) { return i == 0; });
    };

    const auto orderIsSupported = input->desc().dimsOrder() == DimsOrder::NCHW || input->desc().dimsOrder() == DimsOrder::NHWC
                               || input->desc().dimsOrder() == DimsOrder::CHW  || input->desc().dimsOrder() == DimsOrder::HWC;
    VPU_THROW_UNLESS(orderIsSupported, "Current Interpolate supports (N)HWC, (N)CHW data orders only, actual {}", input->desc().dimsOrder());

    //const auto interpolateModeIt = interpModeMap.find(interpolateMode);
    // VPU_THROW_UNLESS(interpolateModeIt != interpModeMap.end(),
    //                  "Current Interpolate supports 'nearest' and 'linear' modes only, actual {}", interpolateMode);
    const auto modeIsSupported = interpolateMode == ngraph::op::v4::Interpolate::InterpolateMode::nearest ||
                                 interpolateMode == ngraph::op::v4::Interpolate::InterpolateMode::linear  ||
                                 interpolateMode == ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx;
    VPU_THROW_UNLESS(modeIsSupported, "Current Interpolate supports 'nearest' and 'linear' modes only, actual {}", interpolateMode);

    auto paramIsSupported = ic == oc;
    VPU_THROW_UNLESS(paramIsSupported, "Current Interpolate does not support resize by channels");
    paramIsSupported = isPadZeros(padsBegin) && isPadZeros(padsEnd);
    VPU_THROW_UNLESS(paramIsSupported, "Current Interpolate does not support paddings");

    if (interpolateMode == ngraph::op::v4::Interpolate::InterpolateMode::nearest) {
        // current "Resample" supports the following "Interpolate" modes only:
        // coordinate_transformation_mode = half_pixel; nearest_mode = round_prefer_ceil;
        // coordinate_transformation_mode = asymmetric; nearest_mode = floor;
        // other "Interpolate" modes are translated to the default ones
        const auto anti = attrs.antialias;
        const auto coordinateTransformationMode = getVpuCoordTransMode(attrs.coordinate_transformation_mode);
        const auto nearestMode = getVpuNearestMode(attrs.nearest_mode);

        // const auto coordModeIt   = coordTransformModeMap.find(coordinateTransformation);
        // const auto nearestModeIt = nearestModeMap.find(near);
        // VPU_THROW_UNLESS(coordModeIt != coordTransformModeMap.end(), "Interpolate stage does not support this coordinate transforation mode");
        // VPU_THROW_UNLESS(nearestModeIt != nearestModeMap.end(), "Interpolate stage does not support this nearest transforation mode");

        _stageBuilder->addResampleNearestStage(model,
                                                interpolate->get_name(),
                                                interpolate,
                                                anti,
                                                coordinateTransformationMode,
                                                nearestMode,
                                                -1.0f,
                                                input,
                                                output);
    } else if (interpolateMode == ngraph::op::v4::Interpolate::InterpolateMode::linear  ||
               interpolateMode == ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx) {
        // current "Interp" supports modes "align_corners" and "asymmetric" only
        // other "Interpolate" modes are translated to the default ones
        const auto coordinateTransformationMode = getVpuCoordTransMode(attrs.coordinate_transformation_mode);
        
        // VPU_THROW_UNLESS(interpolateModeIt != interpModeMap.end(), "Interp stage with name {} does not support this interp mode", _layer->name);
        // VPU_THROW_UNLESS(interpolateModeIt->second == InterpolateMode::Linear || interpolateModeIt->second  == InterpolateMode::LinearOnnx,
                            // "Interp stage supports linear and linear_onnx modes");
        // VPU_THROW_UNLESS(coordModeIt != coordTransformModeMap.end(), "Interp stage does not support this coordinate transforation mode");
        auto mode = attrs.mode;

        _stageBuilder->addInterpStage(model,
                                        interpolate->get_name(),
                                        interpolate,
                                        coordinateTransformationMode == vpu::InterpolateCoordTransMode::AlignCorners,
                                        getVpuInterpolateMode(mode),
                                        coordinateTransformationMode,
                                        input,
                                        output);
    }
}

}  // namespace vpu
