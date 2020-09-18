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

#include "deconvolution_kernel_imad_along_f_tile_bfx.hpp"

#include "kernel_selector_utils.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <string>

namespace kernel_selector {

namespace {
    constexpr size_t simd = 16;
}

ParamsKey DeconvolutionKernel_imad_along_f_tile_bfx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);

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

bool DeconvolutionKernel_imad_along_f_tile_bfx::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o))
        return false;

    auto& params = static_cast<const deconvolution_params&>(p);
    if (params.groups > 1 && params.weights.IFM().v % 4 != 0)
        return false;

    // Consider loosening at the cost of performance
    if (params.groups > 1 && params.weights.OFM().v % simd != 0)
        return false;

    return true;
}

WeightsLayout DeconvolutionKernel_imad_along_f_tile_bfx::GetPreferredWeightsLayout(const deconvolution_params& params) const {
    //                                isv,    osv
    using layout_map_key = std::tuple<size_t, size_t>;
    using layout_map = std::map<layout_map_key, WeightsLayout>;

    layout_map lt_map = {
        {layout_map_key((size_t)4,  (size_t)16), WeightsLayout::g_os_zyx_is_osv16_isv4 },
        {layout_map_key((size_t)16, (size_t)16), WeightsLayout::g_os_zyx_is_osv16_isv16 },
        {layout_map_key((size_t)32, (size_t)16), WeightsLayout::g_os_zyx_is_osv16_isv32 },
        {layout_map_key((size_t)4,  (size_t)32), WeightsLayout::g_os_zyx_is_osv32_isv4 },
        {layout_map_key((size_t)16, (size_t)32), WeightsLayout::g_os_zyx_is_osv32_isv16 },
        {layout_map_key((size_t)32, (size_t)32), WeightsLayout::g_os_zyx_is_osv32_isv32 }};

    auto tile_ifm = GetTileIFM(params);
    auto tile_ofm_simd = GetTileOFM(params) * simd;
    auto key = layout_map_key(tile_ifm, tile_ofm_simd);
    auto it = lt_map.find(key);
    if (it == lt_map.end()) {
        // Params are not valid for this implementation, return anything to allow Validate to reject
        return WeightsLayout::goizyx;
    }
    auto layout = it->second;
    return layout;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_imad_along_f_tile_bfx::SetDefault(const deconvolution_params& params) const {
    auto dispatch = Parent::SetDefault(params);

    auto tile_x = GetTileX(params);
    auto tile_ofm = GetTileOFM(params);
    auto tile_b = GetTileB(params);

    std::vector<size_t> global = {
         CeilDiv(params.output.X().v, tile_x) * params.output.Y().v * params.output.Z().v,
         Align(CeilDiv(params.output.Feature().v, tile_ofm), simd),
         CeilDiv(params.output.Batch().v, tile_b)
    };

    std::vector<size_t> local = { 1, simd, 1 };

    dispatch.gws0 = global[0];
    dispatch.gws1 = global[1];
    dispatch.gws2 = global[2];

    dispatch.lws0 = local[0];
    dispatch.lws1 = local[1];
    dispatch.lws2 = local[2];

    // Currently most optimized for fsv16 formats
    if (params.inputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 || params.inputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16) {
        dispatch.efficiency = FORCE_PRIORITY_7;
    } else {
        dispatch.efficiency = FORCE_PRIORITY_8;
    }

    return dispatch;
}

JitConstants DeconvolutionKernel_imad_along_f_tile_bfx::GetJitConstants(const deconvolution_params& params) const {
    auto jit = Parent::GetJitConstants(params);
    auto tile_ifm = GetTileIFM(params);
    auto tile_x = GetTileX(params);
    auto tile_ofm = GetTileOFM(params);
    auto tile_b = GetTileB(params);

    jit.AddConstant(MakeJitConstant("TILE_IFM", tile_ifm));
    jit.AddConstant(MakeJitConstant("TILE_X", tile_x));
    jit.AddConstant(MakeJitConstant("TILE_OFM", tile_ofm));
    jit.AddConstant(MakeJitConstant("TILE_B", tile_b));
    jit.AddConstant(MakeJitConstant("SIMD", simd));

    auto& in = params.inputs[0];
    auto in_layout = in.GetLayout();

    // Layout specific params
    size_t input_tile_ifm_pitch = 0;
    size_t input_in_tile_batch_pitch = 0;
    size_t zyx_pitch_factor = in.Z().LogicalDimPadded() * in.Y().LogicalDimPadded() * in.X().LogicalDimPadded();

    if (in_layout == DataLayout::b_fs_yx_fsv16 || in_layout == DataLayout::b_fs_zyx_fsv16) {
        if (tile_ifm == 16) {
            input_tile_ifm_pitch = zyx_pitch_factor * 16;
        }
        input_in_tile_batch_pitch = Align(in.Feature().LogicalDimPadded(), 16) * zyx_pitch_factor;
    } else if (in_layout == DataLayout::b_fs_yx_fsv32 || in_layout == DataLayout::b_fs_yx_fsv32) {
        if (tile_ifm == 32) {
            input_tile_ifm_pitch = zyx_pitch_factor * 32;
        }
        input_in_tile_batch_pitch = Align(in.Feature().LogicalDimPadded(), 32) * zyx_pitch_factor;
    } else if (in_layout == DataLayout::bs_fs_yx_bsv16_fsv16 || in_layout == DataLayout::bs_fs_zyx_bsv16_fsv16) {
        if (tile_ifm == 16) {
            input_tile_ifm_pitch = zyx_pitch_factor * 16 * 16;
        }
        input_in_tile_batch_pitch = 16;
    }

    jit.AddConstant(MakeJitConstant("INPUT_VALID_TILE_IFM_PITCH", input_tile_ifm_pitch != 0));
    jit.AddConstant(MakeJitConstant("INPUT_TILE_IFM_PITCH", input_tile_ifm_pitch));
    jit.AddConstant(MakeJitConstant("INPUT_IN_TILE_B_PITCH", input_in_tile_batch_pitch));

    if (params.output.GetLayout() == DataLayout::b_fs_yx_fsv16 || params.output.GetLayout() == DataLayout::b_fs_zyx_fsv16) {
        jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_X_STORE", true));
    } else {
        jit.AddConstant(MakeJitConstant("OUTPUT_NAIVE_STORE", true));
    }

    if (!params.fused_ops.empty()) {
        auto fused_in_dt = GetActivationType(params);
        std::vector<std::string> idx_order;
        if (params.output.Dimentions() <= 4) {
            idx_order = { "(out_b + ob)", "(out_f + of * SIMD)", "out_y", "(out_x + tx)" };
        } else {
            idx_order = { "(out_b + ob)", "(out_f + of * SIMD)", "out_z", "out_y", "(out_x + tx)" };
        }
        auto boundary_check = BoundaryCheck::DISABLED;
        if (params.output.X().v % tile_x != 0
            || params.output.Feature().v % (tile_ofm * simd) != 0
            || params.output.Batch().v % tile_b != 0) {
            boundary_check = BoundaryCheck::ENABLED;
        }
        std::vector<Tensor::DataChannelName> loop_axes = { Tensor::DataChannelName::X };
        if (tile_b != 1) {
            loop_axes.push_back(Tensor::DataChannelName::BATCH);
        } else {
            idx_order[0] = "out_b";
        }

        auto conf = FusedOpsConfiguration{ "",
                                           idx_order,
                                           "dequantized[ob][of][tx]",
                                           fused_in_dt,
                                           1,
                                           LoadType::LT_UNALIGNED,
                                           boundary_check,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X,
                                           loop_axes,
                                           true };

        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

size_t DeconvolutionKernel_imad_along_f_tile_bfx::GetTileIFM(const deconvolution_params& params) const {
    size_t fsv = 4;
    if (params.inputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16
        || params.inputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16
        || params.inputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16
        || params.inputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16) {
        fsv = 16;
    }
    if (params.inputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32
        || params.inputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        fsv = 32;
    }

    auto ifm = params.weights.IFM().v;
    bool grouped = params.groups > 1;
    auto pref_tile_ifm = std::min(fsv, ifm);

    std::vector<size_t> allowed_tile_ifm = { 4, 16, 32 };
    size_t tile_ifm = 1;
    for (auto candidate : allowed_tile_ifm) {
        if (candidate <= pref_tile_ifm
            && (!grouped || ifm % candidate == 0))
            tile_ifm = candidate;
    }
    return tile_ifm;
}

size_t DeconvolutionKernel_imad_along_f_tile_bfx::GetTileOFM(const deconvolution_params& params) const {
    // TODO Loosen divisibility requirement for tile ofm 2
    if (params.weights.OFM().v % (simd * 2) == 0 && params.output.Batch().v % 2 != 0)
        return 2;

    return 1;
}

size_t DeconvolutionKernel_imad_along_f_tile_bfx::GetTileX(const deconvolution_params& params) const {
    constexpr size_t max_tile_x = simd;
    if (params.output.X().v <= max_tile_x)
        return params.output.X().v;

    return max_tile_x;
}

size_t DeconvolutionKernel_imad_along_f_tile_bfx::GetTileB(const deconvolution_params& params) const {
    if (params.output.Batch().v % 2 == 0)
        return 2;

    return 1;
}

}  // namespace kernel_selector
