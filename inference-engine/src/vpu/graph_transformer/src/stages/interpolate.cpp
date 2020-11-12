// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
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

    const auto interpolateMode = _layer->GetParamAsString("mode");

    const auto input = inputs[0];
    const auto output = outputs[0];

    // try to use existing resize layers
    if (input->desc().dimsOrder() == DimsOrder::NCHW || input->desc().dimsOrder() == DimsOrder::NHWC ||
        input->desc().dimsOrder() == DimsOrder::CHW || input->desc().dimsOrder() == DimsOrder::HWC) {
        const auto ic = input->desc().dim(Dim::C);
        const auto in = (input->desc().numDims() == 3) ? 1 : input->desc().dim(Dim::N);

        const auto oc = output->desc().dim(Dim::C);
        const auto on = (output->desc().numDims() == 3) ? 1 : output->desc().dim(Dim::N);

        auto padsBegin = _layer->GetParamAsInts("pads_begin", {});
        auto padsEnd = _layer->GetParamAsInts("pads_end", {});

        const auto isPadZeros = [](const std::vector<int>& pad) {
            return std::all_of(pad.begin(), pad.end(), [](int i) { return i == 0; });
        };

        if (ic == oc && in == 1 && on == 1 && isPadZeros(padsBegin) && isPadZeros(padsEnd)) {
            ie::details::CaselessEq<std::string> cmp;
            if (cmp(interpolateMode, "nearest")) {
                // current "Resample" supports the following "Interpolate" modes only:
                // coordinate_transformation_mode = half_pixel; nearest_mode = round_prefer_ceil;
                // coordinate_transformation_mode = asymmetric; nearest_mode = floor;
                // other "Interpolate" modes are translated to the default ones
                const auto antialias = _layer->GetParamAsBool("antialias", false);
                const auto coordinateTransformation = _layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");
                const auto nearest = _layer->GetParamAsString("nearest_mode", "round_prefer_floor");
                InterpolateCoordTransMode coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                InterpolateNearestMode nearestMode = InterpolateNearestMode::RoundPreferCeil;

                if (cmp(coordinateTransformation, "asymmetric")) {
                    coordinateTransformationMode = InterpolateCoordTransMode::Asymmetric;
                }

                if (cmp(nearest, "round_prefer_floor")) {
                    nearestMode = InterpolateNearestMode::RoundPreferFloor;
                } else if (cmp(nearest, "round_prefer_ceil")) {
                    nearestMode = InterpolateNearestMode::RoundPreferCeil;
                } else if (cmp(nearest, "floor")) {
                    nearestMode = InterpolateNearestMode::Floor;
                }
                _stageBuilder->addResampleNearestStage(model,
                                                       _layer->name,
                                                       _layer,
                                                       antialias,
                                                       coordinateTransformationMode,
                                                       nearestMode,
                                                       -1.0f,
                                                       input,
                                                       output);
            } else if (cmp(interpolateMode, "linear")) {
                // current "Interp" supports modes "align_corners" and "asymmetric" only
                // other "Interpolate" modes are translated to the default ones
                const auto coordinateTransformationMode = _layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");

                _stageBuilder->addInterpStage(model,
                                              _layer->name,
                                              _layer,
                                              cmp(coordinateTransformationMode, "align_corners"),
                                              input,
                                              output);
            } else {
                VPU_THROW_EXCEPTION << "Current Interpolate supports 'nearest' and 'linear' modes only; layer name = " << _layer->name;
            }
        } else {
            VPU_THROW_EXCEPTION << "Current Interpolate does not support paddings, batches, and resize by channels; layer name = " << _layer->name;
        }
    } else {
        VPU_THROW_EXCEPTION << "Current Interpolate supports (N)HWC, (N)CHW data orders only; layer name = " << _layer->name;
    }
}

}  // namespace vpu
