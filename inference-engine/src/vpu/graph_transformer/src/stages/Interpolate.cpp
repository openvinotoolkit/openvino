// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <ie_common.h>
#include <ie_blob.h>
#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

using namespace vpu;
using namespace InferenceEngine;

enum class InterpolateMode {
    nearest,
    linear,
    linear_onnx,
    cubic
};

enum class InterpolateShapeCalcMode {
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

namespace {
class InterpolateStage final : public StageNode {
public:
    using StageNode::StageNode;

    InterpolateStage(const InferenceEngine::CNNLayerPtr& layer);
    ~InterpolateStage() = default;

    struct InterpolateAttrs
    {
        // specifies type of interpolation
        // one of `nearest`, `linear`, `linear_onnx`, `cubic` Required.
        InterpolateMode mode;
        // specifies which input, sizes or scales, is used to calculate an output shape.
        // one of `sizes`, `scales` Required
        InterpolateShapeCalcMode shape_calculation_mode;
        // specifies how to transform the coordinate in the resized tensor to the
        // coordinate in the original tensor. one of `half_pixel`, `pytorch_half_pixel`,
        // `asymmetric`, `tf_half_pixel_for_nn`, `align_corners`
        InterpolateCoordTransMode coordinate_transformation_mode;
        // specifies round mode when `mode == nearest` and is used only when `mode ==
        // nearest`. one of `round_prefer_floor`, `round_prefer_ceil`, `floor`, `ceil`,
        // `simple`
        InterpolateNearestMode nearest_mode;
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
                         InterpolateShapeCalcMode shape_calculation_mode,
                         InterpolateCoordTransMode coordinate_transformation_mode =
                            InterpolateCoordTransMode::half_pixel,
                         InterpolateNearestMode nearest_mode = InterpolateNearestMode::round_prefer_floor,
                         bool antialias = false,
                         std::vector<size_t> pads_begin = {0},
                         std::vector<size_t> pads_end = {0},
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
            , coordinate_transformation_mode(InterpolateCoordTransMode::half_pixel)
            , nearest_mode(InterpolateNearestMode::round_prefer_floor)
            , antialias(false)
            , cube_coeff(-0.75f) {

        }
    } m_attrs;

    void Interpolate();

    // Without 'axes' input.
    // image - Input image
    // output_shape - Output shape of spatial axes
    // scales - Scales of spatial axes, i.e. output_shape / input_shape
    // attrs - Interpolation attributes
    void Interpolate(const DataVector& image,
                const DataVector& output_shape,
                const DataVector& scales,
                const InterpolateAttrs& attrs);

    // With 'axes' input.
    // image - Input image
    // output_shape - Output shape of spatial axes
    // scales - Scales of spatial axes, i.e. output_shape / input_shape
    // axes - Interpolation axes
    // attrs - Interpolation attributes
    void Interpolate(const DataVector& image,
                const DataVector& output_shape,
                const DataVector& scales,
                const DataVector& axes,
                const InterpolateAttrs& attrs);

    const InterpolateAttrs& get_attrs() const { return m_attrs; }
private:
    void execute();
    // nearest neighbor
    void nearestNeighbor(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                         float fx, float fy, float fz, int OD, int OH, int OW);
    void nearestNeighborReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
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
    void cubicInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                    float fx, float fy, int OH, int OW, float a);
    void cubicReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                    float fx, float fy, int OH, int OW, float a);

    int nearestMode(bool isDown, float originalValue);

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
};
// }}


void InterpolateStage::Interpolate() {};

void InterpolateStage::Interpolate(const DataVector& image,
            const DataVector& output_shape,
            const DataVector& scales,
            const InterpolateAttrs& attrs) {
    
}

void InterpolateStage::Interpolate(const DataVector& image,
            const DataVector& output_shape,
            const DataVector& scales,
            const DataVector& axes,
            const InterpolateAttrs& attrs) {
    
}

void InterpolateStage::execute() {

    switch (mode) {
        case InterpolateMode::nearest: {
            nearestNeighbor();
            break;
        }
        case InterpolateMode::linear: {
            linearInterpolation();
            break;
        }
        case InterpolateMode::linear_onnx: {
            linearOnnx();
            break;
        }
        case InterpolateMode::cubic: {
            cubicInterpolation();
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer has unsupported interpolate mode: " << mode;
        }
    }
}

// Idea1: Look ate different methods below and decide wich operation (resample\interp) to choose
// according to the parameters. But isn't it the concept of convert?
// Still have to find a way of getting rid of conversion (?) 

// nearest neighbor
void InterpolateStage::nearestNeighbor(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW) {
    
}
void InterpolateStage::nearestNeighborReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW) {
    
}

// linear
void InterpolateStage::linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    
}

void InterpolateStage::linearReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    
}

// onnx linear
void InterpolateStage::linearOnnx(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                                   float fx, float fy, int OH, int OW) {
    
}
void InterpolateStage::linearOnnxReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                                   float fx, float fy, int OH, int OW) {
    
}

// cubic
std::vector<float> InterpolateStage::getCubicCoef(float a) {
    return std::vector<float>(0);
}

void InterpolateStage::cubicInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                              float fx, float fy, int OH, int OW, float a) {
    
}

void InterpolateStage::cubicReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                              float fx, float fy, int OH, int OW, float a) {
    
}

int InterpolateStage::nearestMode(bool isDown, float originalValue) {
    switch (nearestMode) {
        case InterpolateNearestMode::round_prefer_floor: {
            if (originalValue == (static_cast<int>(originalValue) + 0.5f)) {
                return static_cast<int>(std::floor(originalValue));
            } else {
                return static_cast<int>(std::round(originalValue));
            }
            break;
        }
        case InterpolateNearestMode::round_prefer_ceil: {
            return static_cast<int>(std::round(originalValue));
            break;
        }
        case InterpolateNearestMode::floor: {
            return static_cast<int>(std::floor(originalValue));
            break;
        }
        case InterpolateNearestMode::ceil: {
            return static_cast<int>(std::ceil(originalValue));
            break;
        }
        case InterpolateNearestMode::simple: {
            if (isDown) {
                return static_cast<int>(std::ceil(originalValue));
            } else {
                return static_cast<int>(originalValue);
            }
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' does not support this nearest round mode";
        }
    }
}

std::vector<float> InterpolateStage::getScales() {
    return scales;
}

std::vector<int> InterpolateStage::getAxes() {
    return axes;
}
}  // namespace

Stage StageBuilder::addInterpolateStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const std::string& origin) {
    Stage interpolateStage = model->addNewStage<InterpolateStage>(
        name,
        StageType::Interpolate,
        layer,
        {input},
        {output});
    interpolateStage->attrs().set<std::string>("origin", origin);

    return interpolateStage;
}

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    // ie::details::CaselessEq<std::string> cmp;

    _stageBuilder->addInterpolateStage(model, layer->name, layer, inputs[0], outputs[0], "parseInterpolate");

    // auto stage = model->addNewStage<InterpolateStage>(layer->name, StageType::Interpolate, layer, inputs, outputs);

    // stage->attrs().set<bool>("antialias", layer->GetParamAsInt("antialias", 0));
    // stage->attrs().set<float>("factor", layer->GetParamAsInt("factor", -1.0f));

    // auto method = layer->GetParamAsString("type", "caffe.InterpolateParameter.NEAREST");
    // if (cmp(method, "caffe.InterpolateParameter.NEAREST")) {
    //     stage->attrs().set<InterpolateMode>("type", InterpolateMode::nearest);
    // } else if (cmp(method, "caffe.InterpolateParameter.LINEAR")) {
    //     stage->attrs().set<InterpolateMode>("type", InterpolateMode::linear);
    // } else if (cmp(method, "caffe.InterpolateParameter.CUBIC")) {
    //     stage->attrs().set<InterpolateMode>("type", InterpolateMode::cubic);
    // } else {
    //     VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate type";
    // }

    // auto nnMode = layer->GetParamAsString("nearestMode", "caffe.InterpolateParameter.round_prefer_floor");
    // if (cmp(nnMode, "caffe.InterpolateParameter.round_prefer_floor")) {
    //     stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::round_prefer_floor);
    // } else if (cmp(nnMode, "caffe.InterpolateParameter.round_prefer_ceil")) {
    //     stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::round_prefer_ceil);
    // } else if (cmp(nnMode, "caffe.InterpolateParameter.floor")) {
    //     stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::floor);
    // } else if (cmp(nnMode, "caffe.InterpolateParameter.ceil")) {
    //     stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::ceil);
    // } else if (cmp(nnMode, "caffe.InterpolateParameter.simple")) {
    //     stage->attrs().set<InterpolateNearestMode>("nearestMode", InterpolateNearestMode::simple);
    // } else {
    //     VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate nearest mode variant";
    // }

    // auto shapeCalcMode = layer->GetParamAsString("shapeCalcMode", "caffe.InterpolateParameter.sizes");
    // if (cmp(shapeCalcMode, "caffe.InterpolateParameter.sizes")) {
    //     stage->attrs().set<InterpolateShapeCalcMode>("shapeCalcMode", InterpolateShapeCalcMode::sizes);
    // } else if (cmp(shapeCalcMode, "caffe.InterpolateParameter.scales")) {
    //     stage->attrs().set<InterpolateShapeCalcMode>("shapeCalcMode", InterpolateShapeCalcMode::scales);
    // } else {
    //     VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate shape calculation mode";
    // }

    // auto coordTransMode = layer->GetParamAsString("coordTransMode", "caffe.InterpolateParameter.half_pixel");
    // if (cmp(coordTransMode, "caffe.InterpolateParameter.half_pixel")) {
    //     stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::half_pixel);
    // } else if (cmp(coordTransMode, "caffe.InterpolateParameter.pytorch_half_pixel")) {
    //     stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::pytorch_half_pixel);
    // } else if (cmp(coordTransMode, "caffe.InterpolateParameter.pytorch_half_pixel")) {
    //     stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::pytorch_half_pixel);
    // } else if (cmp(coordTransMode, "caffe.InterpolateParameter.tf_half_pixel_for_nn")) {
    //     stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::tf_half_pixel_for_nn);
    // } else if (cmp(coordTransMode, "caffe.InterpolateParameter.align_corners")) {
    //     stage->attrs().set<InterpolateCoordTransMode>("coordTransMode", InterpolateCoordTransMode::align_corners);
    // } else {
    //     VPU_THROW_EXCEPTION << "Layer with name " << layer->name << " doesn't support this Interpolate coordinate transform mode";
    // }
}
