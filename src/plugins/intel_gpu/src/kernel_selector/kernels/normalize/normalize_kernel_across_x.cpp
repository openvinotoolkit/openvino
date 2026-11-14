// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_kernel_across_x.h"

namespace kernel_selector {
ParamsKey NormalizeKernelAcrossX::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::INT8);
    key.EnableOutputDataType(Datatype::UINT8);
    for (auto layout : {DataLayout::bfyx,
                        DataLayout::yxfb,
                        DataLayout::byxf,
                        DataLayout::b_fs_yx_fsv16,
                        DataLayout::b_fs_yx_fsv32,
                        DataLayout::bs_fs_yx_bsv16_fsv16,
                        DataLayout::bs_fs_yx_bsv32_fsv16,
                        DataLayout::bs_fs_yx_bsv32_fsv32}) {
        key.EnableInputLayout(layout);
        key.EnableOutputLayout(layout);
    }
    key.EnableDifferentTypes();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableNormalizeMode(NormalizeMode::ACROSS_X);
    return key;
}

KernelsData NormalizeKernelAcrossX::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority NormalizeKernelAcrossX::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector