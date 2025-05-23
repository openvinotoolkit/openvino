// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell_and_seq_kernel_bfyx.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey LSTMCellAndSeqKernel_bfyx::GetSupportedKey() const {
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
    return k;
}

KernelsData LSTMCellAndSeqKernel_bfyx::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority LSTMCellAndSeqKernel_bfyx::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}
}  // namespace kernel_selector
