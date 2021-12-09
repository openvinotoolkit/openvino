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

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() <= 4 && inputs.size() >= 1,
                     "Interpolate stage with name {} must have no more than 4 inputs and no less than 1 input, actually provided {} inputs",
                     _layer->name, inputs.size());

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Interpolate stage with name {} must have only 1 output, actually provided {} outputs",
                     _layer->name, outputs.size());

    const auto interpolateMode = _layer->GetParamAsString(g_mode, g_nearest);

    const auto input  = inputs[0];
    const auto output = outputs[0];

    // try to use existing resize layers

    const auto oc = output->desc().dim(Dim::C);
    const auto on = output->desc().dim(Dim::N, 1);

    const auto ic = input->desc().dim(Dim::C);
    const auto in = input->desc().dim(Dim::N, 1);

    VPU_THROW_UNLESS(in == on, "incompatible: input batch=%d, output batch=%d", in, on);

    auto padsBegin = _layer->GetParamAsInts(g_pads_begin, {});
    auto padsEnd   = _layer->GetParamAsInts(g_pads_end, {});

    const auto isPadZeros = [](const std::vector<int>& pad) {
        return std::all_of(pad.begin(), pad.end(), [](int i) { return i == 0; });
    };

    const auto orderIsSupported = input->desc().dimsOrder() == DimsOrder::NCHW || input->desc().dimsOrder() == DimsOrder::NHWC
                               || input->desc().dimsOrder() == DimsOrder::CHW  || input->desc().dimsOrder() == DimsOrder::HWC;
    VPU_THROW_UNLESS(orderIsSupported, "Current Interpolate supports (N)HWC, (N)CHW data orders only, actual {}", input->desc().dimsOrder());

    const auto interpolateModeIt = interpModeMap.find(interpolateMode);
    VPU_THROW_UNLESS(interpolateModeIt != interpModeMap.end(),
                     "Current Interpolate supports 'nearest' and 'linear' modes only, actual {}", interpolateMode);
    const auto modeIsSupported = interpolateModeIt->second == InterpolateMode::Nearest ||
                                 interpolateModeIt->second == InterpolateMode::Linear  ||
                                 interpolateModeIt->second == InterpolateMode::LinearOnnx;
    VPU_THROW_UNLESS(modeIsSupported, "Current Interpolate supports 'nearest' and 'linear' modes only, actual {}", interpolateMode);

    auto paramIsSupported = ic == oc;
    VPU_THROW_UNLESS(paramIsSupported, "Current Interpolate does not support resize by channels");
    paramIsSupported = isPadZeros(padsBegin) && isPadZeros(padsEnd);
    VPU_THROW_UNLESS(paramIsSupported, "Current Interpolate does not support paddings");

    if (interpolateModeIt->second == InterpolateMode::Nearest) {
        // current "Resample" supports the following "Interpolate" modes only:
        // coordinate_transformation_mode = half_pixel; nearest_mode = round_prefer_ceil;
        // coordinate_transformation_mode = asymmetric; nearest_mode = floor;
        // other "Interpolate" modes are translated to the default ones
        const auto anti = _layer->GetParamAsBool(g_antialias, false);
        const auto coordinateTransformation = _layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
        const auto near = _layer->GetParamAsString(g_nearest_mode, g_round_prefer_floor);

        const auto coordModeIt   = coordTransformModeMap.find(coordinateTransformation);
        const auto nearestModeIt = nearestModeMap.find(near);
        VPU_THROW_UNLESS(coordModeIt != coordTransformModeMap.end(), "Interpolate stage does not support this coordinate transforation mode");
        VPU_THROW_UNLESS(nearestModeIt != nearestModeMap.end(), "Interpolate stage does not support this nearest transforation mode");
        auto coordinateTransformationMode = coordModeIt->second;
        auto nearestMode = nearestModeIt->second;

        _stageBuilder->addResampleNearestStage(model,
                                                _layer->name,
                                                _layer,
                                                anti,
                                                coordinateTransformationMode,
                                                nearestMode,
                                                -1.0f,
                                                input,
                                                output);
    } else if (interpolateModeIt->second == InterpolateMode::Linear ||
               interpolateModeIt->second == InterpolateMode::LinearOnnx) {
        // current "Interp" supports modes "align_corners" and "asymmetric" only
        // other "Interpolate" modes are translated to the default ones
        const auto coordinateTransformation = _layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
        const auto coordModeIt  = coordTransformModeMap.find(coordinateTransformation);
        VPU_THROW_UNLESS(interpolateModeIt != interpModeMap.end(), "Interp stage with name {} does not support this interp mode", _layer->name);
        VPU_THROW_UNLESS(interpolateModeIt->second == InterpolateMode::Linear || interpolateModeIt->second  == InterpolateMode::LinearOnnx,
                            "Interp stage supports linear and linear_onnx modes");
        VPU_THROW_UNLESS(coordModeIt != coordTransformModeMap.end(), "Interp stage does not support this coordinate transforation mode");
        auto coordinateTransformationMode = coordModeIt->second;
        auto mode = interpolateModeIt->second;

        _stageBuilder->addInterpStage(model,
                                        _layer->name,
                                        _layer,
                                        coordModeIt->second == InterpolateCoordTransMode::AlignCorners,
                                        mode,
                                        coordinateTransformationMode,
                                        input,
                                        output);
    }
}

}  // namespace vpu
