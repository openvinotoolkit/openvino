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

constexpr auto coordinate_transformation_mode = "coordinate_transformation_mode";
constexpr auto mode                 = "mode";
constexpr auto align_corners        = "align_corners";
constexpr auto asymmetric           = "asymmetric";
constexpr auto linear               = "linear";
constexpr auto half_pixel           = "half_pixel";
constexpr auto linear_onnx          = "linear_onnx";
constexpr auto nearest_mode         = "nearest_mode";
constexpr auto pytorch_half_pixel   = "pytorch_half_pixel";
constexpr auto tf_half_pixel_for_nn = "tf_half_pixel_for_nn";
constexpr auto round_prefer_floor   = "round_prefer_floor";
constexpr auto round_prefer_ceil    = "round_prefer_ceil";
constexpr auto floor_mode           = "floor";
constexpr auto ceil_mode            = "ceil";
constexpr auto simple               = "simple";
constexpr auto antialias            = "antialias";
constexpr auto pads_begin           = "pads_begin";
constexpr auto pads_end             = "pads_end";
constexpr auto nearest              = "nearest";

using namespace InferenceEngine;

namespace vpu {

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() <= 4 && inputs.size() >= 1,
                     "Interpolate stage with name {} must have no more than 4 inputs and no less than 1 input, actually provided {} inputs",
                     _layer->name, inputs.size());

    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Interpolate stage with name {} must have only 1 output, actually provided {} outputs",
                     _layer->name, outputs.size());

    const auto interpolateMode = _layer->GetParamAsString(mode, nearest);

    const auto input = inputs[0];
    const auto output = outputs[0];

    // try to use existing resize layers
    if (input->desc().dimsOrder() == DimsOrder::NCHW || input->desc().dimsOrder() == DimsOrder::NHWC ||
        input->desc().dimsOrder() == DimsOrder::CHW || input->desc().dimsOrder() == DimsOrder::HWC) {
        const auto ic = input->desc().dim(Dim::C);
        const auto in = (input->desc().numDims() == 3) ? 1 : input->desc().dim(Dim::N);

        const auto oc = output->desc().dim(Dim::C);
        const auto on = (output->desc().numDims() == 3) ? 1 : output->desc().dim(Dim::N);

        auto padsBegin = _layer->GetParamAsInts(pads_begin, {});
        auto padsEnd   = _layer->GetParamAsInts(pads_end, {});

        const auto isPadZeros = [](const std::vector<int>& pad) {
            return std::all_of(pad.begin(), pad.end(), [](int i) { return i == 0; });
        };

        ie::details::CaselessEq<std::string> cmp;
        if (ic == oc && in == 1 && on == 1 && isPadZeros(padsBegin) && isPadZeros(padsEnd)) {
            if (cmp(interpolateMode, nearest)) {
                // current "Resample" supports the following "Interpolate" modes only:
                // coordinate_transformation_mode = half_pixel; nearest_mode = round_prefer_ceil;
                // coordinate_transformation_mode = asymmetric; nearest_mode = floor;
                // other "Interpolate" modes are translated to the default ones
                const auto anti = _layer->GetParamAsBool(antialias, false);
                const auto coordinateTransformation = _layer->GetParamAsString(coordinate_transformation_mode, half_pixel);
                const auto near = _layer->GetParamAsString(nearest_mode, round_prefer_floor);
                InterpolateCoordTransMode coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                InterpolateNearestMode nearestMode = InterpolateNearestMode::RoundPreferCeil;

                if (cmp(coordinateTransformation, asymmetric)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::Asymmetric;
                } else if (cmp(coordinateTransformation, half_pixel)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                } else if (cmp(coordinateTransformation, pytorch_half_pixel)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::PytorchHalfPixel;
                } else if (cmp(coordinateTransformation, tf_half_pixel_for_nn)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::TfHalfPixelForNn;
                } else if (cmp(coordinateTransformation, align_corners)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::AlignCorners;
                } else {
                    VPU_THROW_EXCEPTION << "Current Interpolate does not support this coordinate transformation mode; layer name = " << _layer->name;
                }

                if (cmp(near, round_prefer_floor)) {
                    nearestMode = InterpolateNearestMode::RoundPreferFloor;
                } else if (cmp(near, round_prefer_ceil)) {
                    nearestMode = InterpolateNearestMode::RoundPreferCeil;
                } else if (cmp(near, floor_mode)) {
                    nearestMode = InterpolateNearestMode::Floor;
                } else if (cmp(near, ceil_mode)) {
                    nearestMode = InterpolateNearestMode::Ceil;
                } else if (cmp(near, simple)) {
                    nearestMode = InterpolateNearestMode::Simple;
                } else {
                    VPU_THROW_EXCEPTION << "Current Interpolate does not support this nearest mode; layer name = " << _layer->name;
                }

                _stageBuilder->addResampleNearestStage(model,
                                                       _layer->name,
                                                       _layer,
                                                       anti,
                                                       coordinateTransformationMode,
                                                       nearestMode,
                                                       -1.0f,
                                                       input,
                                                       output);
            } else if (cmp(interpolateMode, linear) || cmp(interpolateMode, linear_onnx)) {
                // current "Interp" supports modes "align_corners" and "asymmetric" only
                // other "Interpolate" modes are translated to the default ones
                const auto coordinateTransformation = _layer->GetParamAsString(coordinate_transformation_mode, half_pixel);
                InterpolateCoordTransMode coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                InterpolateMode mode = InterpolateMode::Linear;

                if (cmp(interpolateMode, linear_onnx)) {
                    mode = InterpolateMode::LinearOnnx;
                }

                if (cmp(coordinateTransformation, asymmetric)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::Asymmetric;
                } else if (cmp(coordinateTransformation, half_pixel)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
                } else if (cmp(coordinateTransformation, pytorch_half_pixel)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::PytorchHalfPixel;
                } else if (cmp(coordinateTransformation, tf_half_pixel_for_nn)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::TfHalfPixelForNn;
                } else if (cmp(coordinateTransformation, align_corners)) {
                    coordinateTransformationMode = InterpolateCoordTransMode::AlignCorners;
                } else {
                    VPU_THROW_EXCEPTION << "Current Interpolate does not support this coordinate transformation mode; layer name = " << _layer->name;
                }

                _stageBuilder->addInterpStage(model,
                                              _layer->name,
                                              _layer,
                                              cmp(coordinateTransformation, align_corners),
                                              mode,
                                              coordinateTransformationMode,
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
