// Copyright (c) 2016 Intel Corporation
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


#include "convolution_kernel_yxfb_yxio_b16.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

ParamsKey ConvolutionKernel_yxfb_yxio_b16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
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

std::string ConvolutionKernel_yxfb_yxio_b16::GetKernelName(const convolution_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F32) {
        return kernelName + "_fp32";
    } else {
        return kernelName + "_fp16";
    }
}

namespace {
// how many batches will a single work item compute
size_t GetBatchesPerWorkItem(size_t batch_size, Datatype dataType) {
    if (dataType == Datatype::F16) {
        const uint32_t min_batches_per_wi = 1;
        const uint32_t min_lws = 16;

        if (batch_size % (4 * min_batches_per_wi * min_lws) == 0) {
            return 4 * min_batches_per_wi;  // USE_BLOCK_READ_2 + as_half4
        } else if (batch_size % (2 * min_batches_per_wi * min_lws) == 0) {
            return 2 * min_batches_per_wi;  // USE_BLOCK_READ_1 + as_half2
        } else {
            return min_batches_per_wi;
        }
    } else {
        return 2;
    }
}

size_t GetOfmPerWorkitem(Datatype dataType) {
    if (dataType == Datatype::F16)
        return 16;
    return 8;
}
}  // namespace

ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b16::SetDefault(const convolution_params& arg,
                                                                                int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    const auto filter_ofm_num = arg.weights.OFM().v;
    const auto batch_size = arg.output.Batch().v;
    const uint32_t min_lws = 16;

    const size_t batchesPerWorkItem = GetBatchesPerWorkItem(batch_size, arg.inputs[0].GetDType());
    const size_t ofmPerWorkItem = GetOfmPerWorkitem(arg.inputs[0].GetDType());

    if (arg.inputs[0].GetDType() == Datatype::F16) {
        runInfo.effiency = FORCE_PRIORITY_7;
    } else {
        runInfo.effiency = FORCE_PRIORITY_9;
    }

    runInfo.lws0 = min_lws;
    runInfo.gws0 = filter_ofm_num * batch_size / (ofmPerWorkItem * batchesPerWorkItem);

    return runInfo;
}

bool ConvolutionKernel_yxfb_yxio_b16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }
    const convolution_params& params = static_cast<const convolution_params&>(p);

    const auto filter_ofm_num = params.weights.OFM().v;
    const auto batch_size = params.output.Batch().v;
    const uint32_t min_lws = 16;

    const bool bInputValidated =
        (filter_ofm_num > 0) && (batch_size > 0) && (params.output.Feature().v == filter_ofm_num);

    if (!bInputValidated) {
        return false;
    }

    if (params.inputs[0].GetDType() == Datatype::F16) {
        const uint32_t min_ofm_per_wi = 16;
        const uint32_t min_batches_per_wi = 1;

        const bool bFilterOK =
            filter_ofm_num % min_ofm_per_wi ==
            0;  // Number of output features dividable by minimum number of output features processed inside work item.
        const bool bBatchOK =
            batch_size % (min_batches_per_wi * min_lws) ==
            0;  // Batch size dividable by minimum number of batches processed when smallest local work size is used.

        if (!bFilterOK || !bBatchOK) {
            return false;
        }
    } else {
        if ((filter_ofm_num * batch_size) % min_lws != 0 || batch_size < 32) {  // TODO: check why it's not supported
            return false;
        }
    }

    return true;
}

JitConstants ConvolutionKernel_yxfb_yxio_b16::GetJitConstants(const convolution_params& params,
                                                              const DispatchData& kd) const {
    auto jit = Parent::GetJitConstants(params, kd);

    const auto local_work_group_size = kd.lws0;
    const auto batch_size = params.output.Batch().v;

    if (params.inputs[0].GetDType() == Datatype::F32) {
        // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up
        // by 25%
        // TODO - investigate why is this happening
        if (params.inputs[0].Feature().v > 4) {
            jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
        }
    } else {
        const auto batch_pad_before = params.output.Batch().pad.before;
        const auto feature_pitch = params.output.Feature().pitch;

        if (batch_size >= 64 && (feature_pitch % 2 == 0) && (batch_pad_before % 2 == 0)) {
            jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
        } else if (batch_size >= 32 && (feature_pitch % 2 == 0) && (batch_pad_before % 2 == 0)) {
            jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_1", ""));
        }
    }

    const size_t batchesPerWorkItem = GetBatchesPerWorkItem(batch_size, params.inputs[0].GetDType());
    const size_t ofmPerWorkItem = GetOfmPerWorkitem(params.inputs[0].GetDType());

    jit.AddConstants({
        MakeJitConstant("LOCAL_WORK_GROUP_SIZE", kd.lws0),
        MakeJitConstant("OFM_PER_WORK_ITEM", ofmPerWorkItem),
        MakeJitConstant("BATCHES_PER_WORK_ITEM",
                        batchesPerWorkItem),  // how many batches will a single work item compute
        MakeJitConstant(
            "LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS",
            std::max(batch_size / batchesPerWorkItem / local_work_group_size,
                     static_cast<size_t>(
                         1))),  // how many local work groups we need to compute single element for each batch
        MakeJitConstant(
            "WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",
            batch_size / batchesPerWorkItem),  // how many work items we need to compute single element for each batch
    });

    return jit;
}

KernelsData ConvolutionKernel_yxfb_yxio_b16::GetKernelsData(const Params& params,
                                                            const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector