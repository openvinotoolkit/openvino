// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_kernel_ref.h"

namespace kernel_selector {
ParamsKey BorderKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    return k;
}

KernelsData BorderKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority BorderKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
