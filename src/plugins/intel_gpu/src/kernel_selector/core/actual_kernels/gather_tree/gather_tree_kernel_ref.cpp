// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_kernel_ref.h"

namespace kernel_selector {
KernelsData GatherTreeKernelRef::GetKernelsData(const Params & params, const optional_params & options) const {
    return GetCommonKernelsData(params, options);
}

ParamsKey GatherTreeKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::byxf);

    k.EnableBatching();

    return k;
}

KernelsPriority GatherTreeKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
