// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_ref.h"

namespace kernel_selector {
ParamsKey BroadcastKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    k.EnableDynamicShapesSupport();

    return k;
}

KernelsData BroadcastKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority BroadcastKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
