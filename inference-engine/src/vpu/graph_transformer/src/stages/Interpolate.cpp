// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "Interpolate.hpp"
#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>

MyriadInterpolate::Interpolate() = default;

MyriadInterpolate::Interpolate(const DataVector& image,
            const DataVector& output_shape,
            const DataVector& scales,
            const InterpolateAttrs& attrs) {
    
}

MyriadInterpolate::Interpolate(const DataVector& image,
            const DataVector& output_shape,
            const DataVector& scales,
            const DataVector& axes,
            const InterpolateAttrs& attrs) {
    
}

// nearest neighbor
void MyriadInterpolate::NearestNeighbor(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW) {
    
}
void MyriadInterpolate::NearestNeighborReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW) {
    
}

// linear
void MyriadInterpolate::linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    
}

void MyriadInterpolate::linearReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int ID, int IH, int IW,
                                        float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    
}

// onnx linear
void MyriadInterpolate::linearOnnx(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                                   float fx, float fy, int OH, int OW) {
    
}
void MyriadInterpolate::linearOnnxReference(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                                   float fx, float fy, int OH, int OW) {
    
}

// cubic
std::vector<float> MyriadInterpolate::getCubicCoef(float a) {
    
}

void MyriadInterpolate::cubic(const uint8_t *in_ptr_, uint8_t *out_ptr_, int batch, int channel, int IH, int IW,
                              float fx, float fy, int OH, int OW, float a) {
    
}

inline float MyriadInterpolate::coordTransToInput(int outCoord, float scale, int inShape, int outShape) {
    
}

inline int MyriadInterpolate::nearestRound(float origin, bool isDown) {
    
}

float MyriadInterpolate::getValue(size_t offset, InferenceEngine::Precision prec) {
    
}

void MyriadInterpolate::setValue(size_t offset, float value, InferenceEngine::Precision prec) {
    
}

std::vector<float> MyriadInterpolate::getScales() {
    
}

std::vector<int> MyriadInterpolate::getAxes() {
    
}
