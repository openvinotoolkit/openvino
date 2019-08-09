// Copyright (c) 2019 Intel Corporation
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


#include "fused_conv_eltwise_kernel_imad.h"
#include "common_tools.h"
#include "kernel_selector_utils.h"
#include <vector>

//
// Kernel specific constants
//
#define SIMD_SIZE 16
// Threshold value to calculate the block size.
#define OUT_BLOCK_THRESHOLD 7
// For images 7x7 it's 7 (default), for 14x14 and above it's 14.
#define OUT_BLOCK_WIDTH 7
// For images 7x7 it's 1 (default), for 14x14 and above it's 2.
#define OUT_BLOCK_HEIGHT 1

static void getOutBlock_WH(size_t inW, size_t Stride, size_t Pad, size_t& outW, size_t& outH) {
    outW = OUT_BLOCK_WIDTH * 2;
    outH = OUT_BLOCK_HEIGHT * 2;

    if ((inW <= OUT_BLOCK_THRESHOLD) || (outW * Stride + Pad > SIMD_SIZE)) {
        outW = OUT_BLOCK_WIDTH;
        outH = OUT_BLOCK_HEIGHT;
    }
    if (outW * Stride + Pad > SIMD_SIZE) {
        outW = outH = 4;
    }

    assert(outW * Stride + Pad <= SIMD_SIZE);
}  // getOutBlock_WH

namespace kernel_selector {

ParamsKey fused_conv_eltwise_kernel_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableFusedConvEltwInt8Quantization();
    k.EnableFusedConvEltwOutputCalibration();
    k.DisableTuning();
    k.EnableFusedConvEltwiseRWOutOpt();
    k.EnableEltwiseStride();
    return k;
}

KernelsData fused_conv_eltwise_kernel_imad::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants fused_conv_eltwise_kernel_imad::GetJitConstants(const fused_conv_eltwise_params& params,
                                                             const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    const auto& iDims = input.GetDims();
    const auto& oDims = output.GetDims();
    const auto& weights = params.weights;
    const auto& wDims = weights.GetDims();
    const int iX = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::X);
    const int iY = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::Y);
    const int iF = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::FEATURE);
    const int wOD = WeightsTensor::Channelndex(weights.GetLayout(), Tensor::WeightsChannelName::OFM);
    const int oX = DataTensor::Channelndex(output.GetLayout(), Tensor::DataChannelName::X);
    const int oY = DataTensor::Channelndex(output.GetLayout(), Tensor::DataChannelName::Y);
    mem_consts.AddConstants({
        MakeJitConstant("_IW", iDims[iX].v),
        MakeJitConstant("_IH", iDims[iY].v),
        MakeJitConstant("_ID", RoundUp(iDims[iF].v, 4)),
        MakeJitConstant("IWPAD", iDims[iX].pad.before + iDims[iX].pad.after),
        MakeJitConstant("IHPAD", iDims[iY].pad.before + iDims[iY].pad.after),
        MakeJitConstant("_OW", oDims[oX].v),
        MakeJitConstant("_OH", oDims[oY].v),
        MakeJitConstant("_OD", wDims[wOD].v),
        MakeJitConstant("OWPAD", oDims[oX].pad.before + oDims[oX].pad.after),
        MakeJitConstant("OHPAD", oDims[oY].pad.before + oDims[oY].pad.after),
        MakeJitConstant("SIMD_SIZE", SIMD_SIZE),
        MakeJitConstant("K_HEIGHT", wDims[iY].v),
        MakeJitConstant("K_WIDTH", wDims[iX].v),
        MakeJitConstant("K_STRIDE", params.conv.stride.x),  // X and Y must be equal
        MakeJitConstant("NON_BLOCK_LOAD", 1),
    });

    size_t obw, obh;
    getOutBlock_WH(iDims[iX].v, params.conv.stride.x, iDims[iX].pad.before + iDims[iX].pad.after, obw, obh);
    mem_consts.AddConstants({MakeJitConstant("OUT_BLOCK_WIDTH", obw), MakeJitConstant("OUT_BLOCK_HEIGHT", obh)});

    return mem_consts;
}  // GetJitConstants

fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_imad::SetDefault(
    const fused_conv_eltwise_params& params,
    int) const {
    DispatchData kd;

    const auto& in = params.inputs[0];
    const auto& weights = params.weights;
    const auto& iDims = in.GetDims();
    const auto& wDims = weights.GetDims();
    const int iX = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::X);
    const int iY = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::Y);
    const int iB = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::BATCH);
    const int wOD = WeightsTensor::Channelndex(weights.GetLayout(), Tensor::WeightsChannelName::OFM);

    size_t otw, oth;
    getOutBlock_WH(iDims[iX].v, params.conv.stride.x, iDims[iX].pad.before + iDims[iX].pad.after, otw, oth);

    std::vector<size_t> global = {// globalRange[0] = ((_IW / K_STRIDE) + (OTW - 1)) / OTW;
                                  // number of tiles needed to cover output width
                                  (((iDims[iX].v / params.conv.stride.x) + (otw - 1)) / otw),

                                  // globalRange[1] = ((_IH / K_STRIDE) + (OTH - 1)) / OTH;
                                  // number of tiles needed to cover output height
                                  (((iDims[iY].v / params.conv.stride.y) + (oth - 1)) / oth),

                                  // globalRange[2] = (_OD * _B) + ((_B *_OD) % __WORKGROUP_SIZE);
                                  // round depth range up
                                  ((wDims[wOD].v * iDims[iB].v) + ((wDims[wOD].v * iDims[iB].v) % SIMD_SIZE))};

    std::vector<size_t> local = {1, 1, SIMD_SIZE};

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};
    kd.effiency = FORCE_PRIORITY_1;

    return kd;
}  // SetDefault

bool fused_conv_eltwise_kernel_imad::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    fused_conv_eltwise_params& newParams = *static_cast<fused_conv_eltwise_params*>(kd.params.get());

    if (newParams.conv.stride.x != newParams.conv.stride.y) {
        // Strides must be equial
        return false;
    } else if ((newParams.conv.filterSize.x != m_FilterSizeX) || (newParams.conv.filterSize.y != m_FilterSizeY)) {
        // Kernel does not support such filter size
        return false;
    } else {
        const auto& in = newParams.inputs[0];
        const auto& iDims = in.GetDims();
        const int iX = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::X);
        if (iDims[iX].v % OUT_BLOCK_THRESHOLD != 0) {
            // Input size must be multiple of OUT_BLOCK_THRESHOLD
            return false;
        }
    }

    return true;
}
}  // namespace kernel_selector
