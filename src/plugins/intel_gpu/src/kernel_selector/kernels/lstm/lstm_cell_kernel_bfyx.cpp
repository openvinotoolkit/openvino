// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell_kernel_bfyx.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey LSTMCellKernel_bfyx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

KernelsData LSTMCellKernel_bfyx::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params, false);
}

KernelsPriority LSTMCellKernel_bfyx::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool LSTMCellKernel_bfyx::Validate(const Params& p) const {
    if (!LSTMKernelBase::Validate(p)) {
        return false;
    }
    const lstm_params& lp = static_cast<const lstm_params&>(p);
    auto out =  lp.outputs[0];
    int num_hidden_kernels = static_cast<int>(out.X().v);
    if (num_hidden_kernels % 2 != 0) {
        return false;
    }
    return true;
}
}  // namespace kernel_selector
