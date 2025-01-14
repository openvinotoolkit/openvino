// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft_kernel_opt.h"

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

    std::vector<std::vector<Tensor::DataChannelName>> dimsByGws;

    if (params.transpose_frames) {
        dispatchData.gws = {output.Feature().v, output.Y().v, output.Batch().v};
    } else {
        dispatchData.gws = {output.Y().v, output.Feature().v, output.Batch().v};
    }

    const int wantedThreadsPerBlock = 128;
    const size_t threads = dispatchData.gws[0] < wantedThreadsPerBlock ? dispatchData.gws[0] : wantedThreadsPerBlock;

    dispatchData.lws = {threads, 1, 1};

    //std::cout << dispatchData << std::endl;
    return dispatchData;
}

}  // namespace kernel_selector
