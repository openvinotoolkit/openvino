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


#include "convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

namespace kernel_selector {

ParamsKey Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.DisableTuning();
    return k;
}

KernelsData Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);
    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"",
                                             {"out_b", "16 * j + out_f + get_sub_group_local_id()", "out_y", "out_x"},
                                             "dequantized",
                                             input_dt,
                                             1,
                                             LoadType::FEATURE_SHUFFLE};

        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1::SetDefault(const convolution_params& params,
                                                                           int) const {
    DispatchData kd;
    const auto& output = params.output;

    std::vector<size_t> global = {output.X().v, output.Y().v, output.Feature().v / 32 * output.Batch().v};
    std::vector<size_t> local = {1, 1, SIMD_SIZE};

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};

    kd.efficiency = FORCE_PRIORITY_2;

    return kd;
}  // SetDefault

bool Convolution_kernel_imad_bs_fs_yx_bsv16_fsv16_1x1::Validate(const Params& params,
                                                                const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if ((newParams.filterSize.x != newParams.filterSize.y) || newParams.filterSize.x != 1) {
        // Fitler size needs to be 1x1
        return false;
    }

    if (newParams.stride.x != newParams.stride.y) {
        // Strides must be equal
        return false;
    }
    if (newParams.output.X().v != newParams.output.Y().v) {
        // W and H must be equal
        return false;
    }

    if (newParams.output.Feature().v % 32 != 0) {
        // output feature size must be divided by 32
        return false;
    }

    if (newParams.output.Batch().v % 16 != 0) {
        // batch size must be divided by 16
        return false;
    }

    return true;
}
}  // namespace kernel_selector
