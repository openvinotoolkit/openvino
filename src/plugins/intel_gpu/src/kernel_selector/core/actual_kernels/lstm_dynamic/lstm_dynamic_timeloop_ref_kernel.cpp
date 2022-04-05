// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_dynamic/lstm_dynamic_timeloop_ref_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey LSTM_DynamicTimeloopKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLSTMEltCell();
    k.EnableLSTMGEMMHidden();
    k.EnableLSTMDyanmicOptionalCellOutput();
    k.EnableLSTMDyanmicOptionalHiddenOutput();
    return k;
}

KernelsData LSTM_DynamicTimeloopKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority LSTM_DynamicTimeloopKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
