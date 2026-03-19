// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bevpool_v2_kernel_ref.h"

namespace kernel_selector {

ParamsKey BevPoolV2KernelRef::GetSupportedKey() const {
    ParamsKey key;

    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableInputDataType(Datatype::INT32);
    key.EnableInputDataType(Datatype::UINT32);
    key.EnableInputDataType(Datatype::INT64);

    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);

    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableInputLayout(DataLayout::bfzyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfzyx);

    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDifferentTypes();
    key.EnableDynamicShapesSupport();
    return key;
}

KernelsData BevPoolV2KernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority BevPoolV2KernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

}  // namespace kernel_selector
