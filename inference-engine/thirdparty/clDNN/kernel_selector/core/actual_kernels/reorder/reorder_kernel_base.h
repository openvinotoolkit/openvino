// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();

        if (winograd) {
            k.EnableWinogradReorder();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reorder_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reorder_optional_params : optional_params {
    reorder_optional_params() : optional_params(KernelType::REORDER) {}
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

    virtual ParamsKey GetParamsKey() const {
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
    virtual bool Validate(const Params&, const optional_params&) const { return true; };
    KernelsData GetCommonKernelsData(const reorder_weights_params& params,
                                     const optional_params&) const;
    KernelsData GetCommonKernelsData(const reorder_params& params, const optional_params&) const;
};
}  // namespace kernel_selector
