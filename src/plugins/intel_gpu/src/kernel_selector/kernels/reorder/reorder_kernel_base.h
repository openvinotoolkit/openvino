// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reorder_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reorder_params : public base_params {
    reorder_params() : base_params(KernelType::REORDER),
    winograd_input_offset_x(0), winograd_input_offset_y(0), winograd_nr_tiles_x(0) {}

    MeanSubtractMode mode = MeanSubtractMode::NONE;
    MeanOp mean_op = MeanOp::SUB;
    std::vector<float> meanValues;
    DataTensor mean;
    uint32_t winograd_input_offset_x;
    uint32_t winograd_input_offset_y;
    uint32_t winograd_nr_tiles_x;
    bool winograd = false;
    bool has_padded_output = false;
    bool surface_input = false;
    bool truncate = false;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();

        if (surface_input) {
            k.EnableSurfaceInputSupport();
        }

        if (winograd) {
            k.EnableWinogradReorder();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reorder_fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reorder_fuse_params : fuse_params {
    DataLayout input_layout;
    DataLayout output_layout;

    reorder_fuse_params(DataLayout input_layout, DataLayout output_layout) :
        fuse_params(KernelType::REORDER), input_layout(input_layout), output_layout(output_layout) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reorder_weights_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reorder_weights_params : public Params {
    reorder_weights_params() : Params(KernelType::REORDER, "") {}

    WeightsTensor input;
    WeightsTensor output;
    bool winograd = false;
    bool rotate_180 = false;

    ParamsKey GetParamsKey() const override {
        ParamsKey k;
        k.EnableInputWeightsType(input.GetDType());
        k.EnableOutputWeightsType(output.GetDType());
        k.EnableInputWeightsLayout(input.GetLayout());
        k.EnableOutputWeightsLayout(output.GetLayout());

        if (input.PitchesDifferFromLogicalDims() || output.PitchesDifferFromLogicalDims()) {
            k.EnableTensorPitches();
        }

        if (input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0) {
            k.EnableTensorOffset();
        }

        if (winograd) {
            k.EnableWinogradReorder();
        }

        if (rotate_180) {
            k.EnableRotateReorder();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ReorderKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReorderKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ReorderKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    virtual JitConstants GetJitConstants(const reorder_weights_params& params) const;
    virtual JitConstants GetJitConstants(const reorder_params& params) const;
    virtual DispatchData SetDefault(const reorder_weights_params& params) const;
    virtual DispatchData SetDefault(const reorder_params& params) const;
    bool Validate(const Params&) const override { return true; }
    KernelsData GetCommonKernelsData(const reorder_weights_params& params) const;
    KernelsData GetCommonKernelsData(const reorder_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
