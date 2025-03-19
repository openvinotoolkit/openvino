// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft_kernel_opt.h"

const size_t FREQ_PER_BLOCK = 256;
const size_t STATIC_MAX_X_I_BUFFER = 2048;
const size_t THREADS_PER_BLOCK = 256;
namespace kernel_selector {
ParamsKey STFTKernelOpt::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);

    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants STFTKernelOpt::GetJitConstants(const STFT_params& params) const {
    JitConstants jit = STFTKernelBase::GetJitConstants(params);

    jit.AddConstants({MakeJitConstant("FREQ_PER_BLOCK", FREQ_PER_BLOCK)});
    jit.AddConstants({MakeJitConstant("STATIC_MAX_X_I_BUFFER", STATIC_MAX_X_I_BUFFER)});

    const auto xiMaxBuffer = params.is_shape_agnostic ? "STATIC_MAX_X_I_BUFFER" : "INPUT1_SIZE_X";
    jit.AddConstants({MakeJitConstant("SHARED_X_I_BUFFER_SIZE", xiMaxBuffer)});

    return jit;
}

KernelsData STFTKernelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority STFTKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

CommonDispatchData STFTKernelOpt::CalcLaunchConfig(const STFT_params& params) const {
    CommonDispatchData dispatchData;
    const auto& output = params.outputs.front();

    OPENVINO_ASSERT(output.Dimentions() == 4);
    OPENVINO_ASSERT(output.X().v == 2);

    const size_t freqSize = params.transpose_frames ? output.Feature().v : output.Y().v;
    const size_t blocksPerFreq = (freqSize + FREQ_PER_BLOCK - 1) / FREQ_PER_BLOCK;

    const size_t framesSize = params.transpose_frames ? output.Y().v : output.Feature().v;
    const size_t batchSize = output.Batch().v;

    dispatchData.lws = {1, THREADS_PER_BLOCK};
    dispatchData.gws = {batchSize, framesSize * THREADS_PER_BLOCK * blocksPerFreq};

    return dispatchData;
}

bool STFTKernelOpt::Validate(const Params& p) const {
    if (STFTKernelBase::Validate(p) == false)
        return false;

    const auto& params = static_cast<const STFT_params&>(p);
    const auto windowSize = params.inputs[1].LogicalSize();

    if (params.is_shape_agnostic && windowSize > STATIC_MAX_X_I_BUFFER)
        return false;

    return true;
}

DeviceFeaturesKey STFTKernelOpt::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();

    return k;
}

}  // namespace kernel_selector
