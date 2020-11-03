//
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
//

#include "deconvolution_kernel_b_fs_zyx_fsv16_dw.h"
#include "kernel_selector_utils.h"

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;
static const float max_reg_pressure = 3.f / 4.f;

float DeconvolutionKernel_b_fs_zyx_fsv16_dw::EstimateRegPressure(const deconvolution_params& params,
                                                                 const dispatch_params& d_params) const {
    size_t usage_bytes = 0;

    usage_bytes += d_params.block_size_x * BytesPerElement(GetAccumulatorType(params));

    if (d_params.preload_weights == weights_preload::all) {
        usage_bytes += params.weights.X().v * params.weights.Y().v * params.weights.Z().v * BytesPerElement(params.weights.GetDType());
    } else if (d_params.preload_weights == weights_preload::line) {
        usage_bytes += params.weights.X().v * BytesPerElement(params.weights.GetDType());
    } else {
        usage_bytes += BytesPerElement(params.weights.GetDType());
    }

    if (d_params.preload_input == input_preload::line) {
        size_t input_line_size = CeilDiv(d_params.block_size_x + params.weights.X().v - 1, params.stride.x);
        usage_bytes += input_line_size * BytesPerElement(params.inputs[0].GetDType());
    } else {
        usage_bytes += BytesPerElement(params.inputs[0].GetDType());
    }

    constexpr size_t register_num = 128;
    constexpr size_t register_bytes = 32;
    constexpr size_t max_register_bytes = register_num * register_bytes;

    return static_cast<float>(usage_bytes * sub_group_size) / static_cast<float>(max_register_bytes);
}

DeconvolutionKernel_b_fs_zyx_fsv16_dw::dispatch_params
DeconvolutionKernel_b_fs_zyx_fsv16_dw::GetDispatchParams(const deconvolution_params& params) const {
    std::vector<dispatch_params> ordered_params;
    if (params.inputs[0].GetDType() == Datatype::F16 || params.inputs[0].GetDType() == Datatype::F32) {
        ordered_params = {
            // Preload weights
            dispatch_params{8, input_preload::none, weights_preload::all},
            dispatch_params{4, input_preload::none, weights_preload::all},
            dispatch_params{2, input_preload::none, weights_preload::all},
            dispatch_params{1, input_preload::none, weights_preload::all},
            // No preloading
            dispatch_params{8, input_preload::none, weights_preload::none},
            dispatch_params{4, input_preload::none, weights_preload::none},
            dispatch_params{2, input_preload::none, weights_preload::none},
            dispatch_params{1, input_preload::none, weights_preload::none},
        };
    } else {
        ordered_params = {
            dispatch_params{16, input_preload::line, weights_preload::line},
            dispatch_params{8,  input_preload::line, weights_preload::line},
            dispatch_params{4,  input_preload::line, weights_preload::line},
            dispatch_params{16, input_preload::line, weights_preload::none},
            dispatch_params{8,  input_preload::line, weights_preload::none},
            dispatch_params{4,  input_preload::line, weights_preload::none},
            dispatch_params{2,  input_preload::line, weights_preload::line},
            dispatch_params{2,  input_preload::line, weights_preload::none},
            dispatch_params{1,  input_preload::line, weights_preload::none},
            dispatch_params{1,  input_preload::none, weights_preload::none},
        };
    }

    dispatch_params best_params = dispatch_params{ 1,  input_preload::none, weights_preload::none };

    for (auto& d_params : ordered_params) {
        bool good_block_size_x = params.output.X().v % d_params.block_size_x == 0 || params.output.X().v > d_params.block_size_x * 3;
        bool good_reg_pressure = EstimateRegPressure(params, d_params) <= max_reg_pressure;
        // No support for no input preload and weights line preload in kernel
        bool good_preloads = !(d_params.preload_input == input_preload::none && d_params.preload_weights == weights_preload::line);
        // At least one input preload
        bool full_input_preload = d_params.preload_input != input_preload::line ||
                                  CeilDiv(d_params.block_size_x + params.filterSize.x - 1, params.stride.x) <= params.inputs[0].X().v;

        if (good_block_size_x && good_reg_pressure && good_preloads && full_input_preload) {
            best_params = d_params;
            break;
        }
    }

    return best_params;
}

ParamsKey DeconvolutionKernel_b_fs_zyx_fsv16_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableAllInputWeightsType();
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableGroupedConvolution();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_b_fs_zyx_fsv16_dw::SetDefault(const deconvolution_params& params) const {
    DispatchData dispatchData = DeconvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = CeilDiv(x, GetDispatchParams(params).block_size_x) * y * z;
    dispatchData.gws[1] = Align(f, feature_block_size);
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = sub_group_size;
    dispatchData.lws[2] = 1;

    dispatchData.efficiency = FORCE_PRIORITY_2;

    return dispatchData;
}

bool DeconvolutionKernel_b_fs_zyx_fsv16_dw::Validate(const Params& p, const optional_params& o) const {
    if (!DeconvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const deconvolution_params&>(p);

    if (params.groups == 1)
        return false;

    if (params.weights.IFM().v != 1 || params.weights.OFM().v != 1)
        return false;

    // Check that padding features doesn't miss-align the blocks
    if (params.inputs[0].Feature().pad.before % feature_block_size != 0 || params.output.Feature().pad.before % feature_block_size != 0)
        return false;

    return true;
}

JitConstants DeconvolutionKernel_b_fs_zyx_fsv16_dw::GetJitConstants(const deconvolution_params& params) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params);

    auto dp = GetDispatchParams(params);
    auto& block_size_x = dp.block_size_x;

    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", block_size_x));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", params.output.Feature().v % feature_block_size));
    }
    jit.AddConstant(MakeJitConstant("INPUT_BLOCK_SIZE_X", CeilDiv(block_size_x + params.filterSize.x - 1, params.stride.x)));
    jit.AddConstant(MakeJitConstant("PRELOAD_INPUT_LINE", dp.preload_input == input_preload::line));
    jit.AddConstant(MakeJitConstant("PRELOAD_WEIGHTS", dp.preload_weights == weights_preload::all));
    jit.AddConstant(MakeJitConstant("PRELOAD_WEIGHTS_LINE", dp.preload_weights == weights_preload::line));

    if (!params.fused_ops.empty()) {
        auto fused_dt = GetActivationType(params);
        std::vector<std::string> idx_order;
        if (params.output.Dimentions() <= 4) {
            idx_order = {"b", "fg", "y", "x"};
        } else {
            idx_order = { "b", "fg", "z", "y", "x" };
        }
        auto boundary_check = BoundaryCheck::ENABLED;
        if (params.output.Feature().v % feature_block_size == 0 && params.output.X().v % block_size_x == 0) {
            boundary_check = BoundaryCheck::DISABLED;
        }
        FusedOpsConfiguration conf = {
            "",
            idx_order,
            "dequantized",
            fused_dt,
            block_size_x,
            LoadType::LT_ALIGNED_READ,
            boundary_check,
            IndexType::TENSOR_COORD,
            Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

}  // namespace kernel_selector
