// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

#include <map>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resample_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct resample_params : public base_params {
    resample_params() : base_params(KernelType::RESAMPLE) {}

    std::vector<int32_t> pads_begin = {};
    std::vector<int32_t> pads_end = {};
    ResampleType resampleType = ResampleType::NEAREST_NEIGHBOR;
    CoordinateTransformationMode coordTransMode = CoordinateTransformationMode::HALF_PIXEL;
    NearestMode nearestMode = NearestMode::ROUND_PREFER_FLOOR;
    ShapeCalculationMode shapeCalculationMode = ShapeCalculationMode::SIZES;
    uint32_t antialias = 0;
    float cube_coeff = -0.75f;
    using AxesAndScales = std::map<InterpolateAxis, float>;
    std::vector<InterpolateAxis> axes;
    std::vector<float> scales;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        k.EnableReampleType(resampleType);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ResampleKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ResampleKernelBase : public KernelBaseOpenCL {
public:
    using DispatchData = CommonDispatchData;
    using KernelBaseOpenCL::KernelBaseOpenCL;

    virtual ~ResampleKernelBase() {}

protected:
    bool Validate(const Params& p) const override;
    virtual DispatchData SetDefault(const resample_params& arg) const;
    virtual JitConstants GetJitConstants(const resample_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    size_t GetFeatureBlockSize(const resample_params& params) const;
    virtual Datatype GetAccumulatorType(const resample_params& params) const;
};
}  // namespace kernel_selector
