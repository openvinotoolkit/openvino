// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_kernel_ref.h"

namespace kernel_selector {
KernelsData GatherTreeKernelRef::GetKernelsData(const Params & params) const {
    return GetCommonKernelsData(params);
}

ParamsKey GatherTreeKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT32);

    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);

    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::byxf);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv4_fsv4);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv4_fsv4);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv8_fsv4);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv8_fsv4);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv8_fsv2);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv8_fsv2);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv4_fsv2);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv4_fsv2);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

    k.EnableTensorPitches();
    k.EnableBatching();

    return k;
}

KernelsPriority GatherTreeKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
