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


#include "convolution_kernel_yxfb_yxio_b8.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_yxfb_yxio_b8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDilation();
    k.EnableSubGroup();
    return k;
}

namespace {
size_t GetOfmPerWorkitem(size_t filterOfmNum, size_t batchSize, size_t local_work_size) {
    if (((filterOfmNum * batchSize) / 16) % local_work_size) {
        return 8;
    } else {
        return 16;
    }
}
}  // namespace

ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b8::SetDefault(const convolution_params& arg,
                                                                               int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(arg, autoTuneIndex);

    const auto filterOfmNum = arg.weights.OFM().v;
    const auto batchSize = arg.output.Batch().v;

    dispatchData.lws[0] = batchSize == 8 ? 8 : 16;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    size_t ofmPerWorkItem = GetOfmPerWorkitem(filterOfmNum, batchSize, dispatchData.lws[0]);

    dispatchData.gws[0] = filterOfmNum * batchSize / ofmPerWorkItem;

    dispatchData.efficiency = FORCE_PRIORITY_9;

    return dispatchData;
}

bool ConvolutionKernel_yxfb_yxio_b8::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const convolution_params& params = static_cast<const convolution_params&>(p);

    if (!CheckPitchForSplitOnly(params)) {
        return false;
    }

    const auto filterOfmNum = params.weights.OFM().v;
    const auto batchSize = params.output.Batch().v;

    const bool bInputValidated = (filterOfmNum > 0) && (batchSize > 0) && (params.output.Feature().v == filterOfmNum);

    if (!bInputValidated) {
        return false;
    }

    const uint32_t lws0 = batchSize == 8 ? 8 : 16;

    if ((filterOfmNum * batchSize) % lws0 != 0 || batchSize > 16 || batchSize == 1) {
        return false;
    }

    if (params.output.PitchesDifferFromLogicalDims())
        return false;

    return true;
}

JitConstants ConvolutionKernel_yxfb_yxio_b8::GetJitConstants(const convolution_params& params,
                                                             const DispatchData& dispatchData) const {
    JitConstants jits = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    size_t ofmPerWorkItem = GetOfmPerWorkitem(params.weights.OFM().v, params.output.Batch().v, dispatchData.lws[0]);

    jits.AddConstant(MakeJitConstant("OFM_PER_WORK_ITEM", ofmPerWorkItem));
    jits.AddConstant(MakeJitConstant("LOCAL_WORK_GROUP_SIZE", dispatchData.lws[0]));

    return jits;
}

KernelsData ConvolutionKernel_yxfb_yxio_b8::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector