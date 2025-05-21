// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

CommonDispatchData GridSampleKernelRef::CalcDispatch(const grid_sample_params& kernel_params) const {
    CommonDispatchData dispatch_data;
    const auto& output = kernel_params.outputs.front();

    dispatch_data.gws = {output.Batch().v * output.Feature().v, output.Y().v, output.X().v};
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo);

    return dispatch_data;
}

ParamsKey GridSampleKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableAllInputDataType();
    key.EnableAllOutputDataType();
    key.EnableDifferentTypes();
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    key.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    key.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    key.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    key.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    key.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    key.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    key.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    key.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    key.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

}  // namespace kernel_selector
