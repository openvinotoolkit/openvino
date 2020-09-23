// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <vpu/frontend/frontend.hpp>
#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

namespace MyriadPlugin {
enum class InterpolateMode {
    nearest,
    linear,
    linear_onnx,
    cubic
};

enum class ShapeCalcMode {
    sizes,
    scales
};

enum class InterpolateCoordTransMode {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode {
    round_prefer_floor,
    round_prefer_ceil,
    floor,
    ceil,
    simple
};

class MyriadInterpolate {
private:
    // nearest neighbor
    void NearestNeighbor(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                         float fx, float fy, float fz, int OD, int OH, int OW);
    void NearestNeighborReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                         float fx, float fy, float fz, int OD, int OH, int OW);

    // onnx linear
    void linearOnnx(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                    float fx, float fy, int OH, int OW);
    void linearOnnxReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                    float fx, float fy, int OH, int OW);

    // linear
    void linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                    float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);
    void linearReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channelC, int ID, int IH, int IW,
                    float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);

    // cubic
    std::vector<float> getCubicCoef(float a);
    void cubic(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                    float fx, float fy, int OH, int OW, float a);
    void cubicReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                    float fx, float fy, int OH, int OW, float a);
    float getValue(size_t offset, InferenceEngine::Precision precision);
    void setValue(size_t offset, float value, InferenceEngine::Precision precision);

    std::vector<float> getScales();
    std::vector<int> getAxes();

    InterpolateMode mode = InterpolateMode::nearest;
    InterpolateCoordTransMode coordTransMode = InterpolateCoordTransMode::half_pixel;
    InterpolateNearestMode nearestMode       = InterpolateNearestMode::round_prefer_floor;

    bool antialias  = false;
    bool hasPad     = false;
    bool hasSpecifiedAxis = false;
    float cubeCoeff = -0.75;

    std::vector<int> padBegin;
    std::vector<int> padEnd;
    std::vector<int> axes = {0, 1, 2, 3};
    std::vector<float> scales = {1.f, 1.f, 2.f, 2.f};

    SizeVector dstDim;
    SizeVector srcDim;
    SizeVector srcPad;

    InferenceEngine::Precision inputPrecision, outputPrecision;
    size_t srcDataSize, dstDataSize;

public:
    MyriadInterpolate(const InferenceEngine::CNNLayerPtr& layer);
    ~MyriadInterpolate() override = default;

    struct InterpolateAttrs
    {
        // specifies type of interpolation
        // one of `nearest`, `linear`, `linear_onnx`, `cubic` Required.
        InterpolateMode mode;
        // specifies which input, sizes or scales, is used to calculate an output shape.
        // one of `sizes`, `scales` Required
        ShapeCalcMode shape_calculation_mode;
        // specifies how to transform the coordinate in the resized tensor to the
        // coordinate in the original tensor. one of `half_pixel`, `pytorch_half_pixel`,
        // `asymmetric`, `tf_half_pixel_for_nn`, `align_corners`
        CoordinateTransformMode coordinate_transformation_mode;
        // specifies round mode when `mode == nearest` and is used only when `mode ==
        // nearest`. one of `round_prefer_floor`, `round_prefer_ceil`, `floor`, `ceil`,
        // `simple`
        NearestMode nearest_mode;
        // a flag that specifies whether to perform anti-aliasing. default is `false`
        bool antialias;
        // specify the number of pixels to add to the beginning of the image being
        // interpolated.
        std::vector<size_t> pads_begin;
        // specify the number of pixels to add to the end of the image being
        // interpolated.
        std::vector<size_t> pads_end;
        // specifies the parameter *a* for cubic interpolation
        // used only when `mode == cubic`
        double cube_coeff;

        InterpolateAttrs(InterpolateMode mode,
                         ShapeCalcMode shape_calculation_mode,
                         CoordinateTransformMode coordinate_transformation_mode =
                            CoordinateTransformMode::half_pixel,
                         NearestMode nearest_mode = NearestMode::round_prefer_floor,
                         bool antialias = false,
                         std::vector<size_t> pads_begin,
                         std::vector<size_t> pads_end,
                         double cube_coeff = -0.75)
            : mode(mode)
            , shape_calculation_mode(shape_calculation_mode)
            , coordinate_transformation_mode(coordinate_transformation_mode)
            , nearest_mode(nearest_mode)
            , antialias(antialias)
            , pads_begin(pads_begin)
            , pads_end(pads_end)
            , cube_coeff(cube_coeff) {

        }

        InterpolateAttrs()
            : mode(InterpolateMode::nearest)
            , coordinate_transformation_mode(CoordinateTransformMode::half_pixel)
            , nearest_mode(NearestMode::round_prefer_floor)
            , antialias(false)
            , cube_coeff(-0.75f) {

        }
    };

    Interpolate() = default;

    // Without 'axes' input.
    // image - Input image
    // output_shape - Output shape of spatial axes
    // scales - Scales of spatial axes, i.e. output_shape / input_shape
    // attrs - Interpolation attributes
    Interpolate(const DataVector& image,
                const DataVector& output_shape,
                const DataVector& scales,
                const InterpolateAttrs& attrs);

    // With 'axes' input.
    // image - Input image
    // output_shape - Output shape of spatial axes
    // scales - Scales of spatial axes, i.e. output_shape / input_shape
    // axes - Interpolation axes
    // attrs - Interpolation attributes
    Interpolate(const DataVector& image,
                const DataVector& output_shape,
                const DataVector& scales,
                const DataVector& axes,
                const InterpolateAttrs& attrs);

    const InterpolateAttrs& get_attrs() const { return m_attrs; }
};
}  // namespace MyriadPlugin
