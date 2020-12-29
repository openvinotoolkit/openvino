// Copyright (C) 2020 Intel Corporation
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

    const auto input = inputs[0];
    const auto output = outputs[0];

    // try to use existing resize layers
    if (input->desc().dimsOrder() == DimsOrder::NCHW || input->desc().dimsOrder() == DimsOrder::NHWC ||
        input->desc().dimsOrder() == DimsOrder::CHW || input->desc().dimsOrder() == DimsOrder::HWC) {
        const auto ic = input->desc().dim(Dim::C);
        const auto in = (input->desc().numDims() == 3) ? 1 : input->desc().dim(Dim::N);

        const auto oc = output->desc().dim(Dim::C);
        const auto on = (output->desc().numDims() == 3) ? 1 : output->desc().dim(Dim::N);

        auto padsBegin = _layer->GetParamAsInts(g_pads_begin, {});
        auto padsEnd   = _layer->GetParamAsInts(g_pads_end, {});

        const auto isPadZeros = [](const std::vector<int>& pad) {
            return std::all_of(pad.begin(), pad.end(), [](int i) { return i == 0; });
        };

        ie::details::CaselessEq<std::string> cmp;
        if (ic == oc && in == 1 && on == 1 && isPadZeros(padsBegin) && isPadZeros(padsEnd)) {
            if (cmp(interpolateMode, g_nearest)) {
                // current "Resample" supports the following "Interpolate" modes only:
                // coordinate_transformation_mode = half_pixel; nearest_mode = round_prefer_ceil;
                // coordinate_transformation_mode = asymmetric; nearest_mode = floor;
                // other "Interpolate" modes are translated to the default ones
                const auto anti = _layer->GetParamAsBool(g_antialias, false);
                const auto coordinateTransformation = _layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
                const auto near = _layer->GetParamAsString(g_nearest_mode, g_round_prefer_floor);
                auto coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                auto nearestMode = InterpolateNearestMode::RoundPreferCeil;

                nearestMode = nearestModeMap.at(near);
                coordinateTransformationMode = coordTransformModeMap.at(coordinateTransformation);

                _stageBuilder->addResampleNearestStage(model,
                                                       _layer->name,
                                                       _layer,
                                                       anti,
                                                       coordinateTransformationMode,
                                                       nearestMode,
                                                       -1.0f,
                                                       input,
                                                       output);
            } else if (cmp(interpolateMode, g_linear) || cmp(interpolateMode, g_linear_onnx)) {
                // current "Interp" supports modes "align_corners" and "asymmetric" only
                // other "Interpolate" modes are translated to the default ones
                const auto coordinateTransformation = _layer->GetParamAsString(g_coordinate_transformation_mode, g_half_pixel);
                auto coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                auto mode = InterpolateMode::Linear;

                if (cmp(interpolateMode, g_linear_onnx)) {
                    mode = InterpolateMode::LinearOnnx;
                }
                coordinateTransformationMode = coordTransformModeMap.at(coordinateTransformation);

                _stageBuilder->addInterpStage(model,
                                              _layer->name,
                                              _layer,
                                              cmp(coordinateTransformation, g_align_corners),
                                              mode,
                                              coordinateTransformationMode,
                                              input,
                                              output);
            } else {
                VPU_THROW_FORMAT("Current Interpolate supports 'nearest' and 'linear' modes only");
            }
        } else {
            VPU_THROW_FORMAT("Current Interpolate does not support paddings, batches, and resize by channels");
        }
    } else {
        VPU_THROW_FORMAT("Current Interpolate supports (N)HWC, (N)CHW data orders only");
    }
}

}  // namespace vpu
