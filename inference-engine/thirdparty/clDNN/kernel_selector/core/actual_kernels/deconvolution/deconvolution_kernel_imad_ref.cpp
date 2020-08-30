// Copyright (c) 2020 Intel Corporation
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

#include "deconvolution_kernel_imad_ref.hpp"

#include "kernel_selector_utils.h"

#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey DeconvolutionKernel_imad_ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableAllOutputLayout();

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableGroupedConvolution();

    return k;
}

WeightsLayout DeconvolutionKernel_imad_ref::GetPreferredWeightsLayout(const deconvolution_params&) const {
    return WeightsLayout::g_os_zyx_is_osv32_isv4;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_imad_ref::SetDefault(const deconvolution_params& params) const {
    auto dispatch = Parent::SetDefault(params);

    std::vector<size_t> global = {
         params.output.Feature().v,
         params.output.X().v * params.output.Y().v * params.output.Z().v,
         params.output.Batch().v
    };

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    dispatch.gws0 = global[0];
    dispatch.gws1 = global[1];
    dispatch.gws2 = global[2];

    dispatch.lws0 = local[0];
    dispatch.lws1 = local[1];
    dispatch.lws2 = local[2];

    dispatch.efficiency = FORCE_PRIORITY_9;

    return dispatch;
}

JitConstants DeconvolutionKernel_imad_ref::GetJitConstants(const deconvolution_params& params) const {
    auto jit = Parent::GetJitConstants(params);
    auto tile_ifm = GetTileIFM(params);

    jit.AddConstant(MakeJitConstant("TILE_IFM", tile_ifm));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.output.Dimentions() <= 4) {
            idx_order = { "out_b", "out_f", "out_y", "out_x" };
        } else {
            idx_order = { "out_b", "out_f", "out_z", "out_y", "out_x" };
        }
        auto conf = FusedOpsConfiguration{ "", idx_order, "dequantized", GetActivationType(params), 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

size_t DeconvolutionKernel_imad_ref::GetTileIFM(const deconvolution_params&) const {
    return 4;
}


}  // namespace kernel_selector
