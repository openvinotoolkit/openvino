// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling_kernel_ps_ref.h"

namespace kernel_selector {

ParamsKey PSROIPoolingKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolType(PoolType::BILINEAR);
    k.EnablePoolType(PoolType::DEFORMABLE_BILINEAR);
    k.EnablePositionSensitivePooling();
    return k;
}

JitConstants PSROIPoolingKernelRef::GetJitConstants(const roi_pooling_params& rp) const {
    JitConstants jit = ROIPoolingKernelBase::GetJitConstants(rp);

    jit.AddConstants({
        MakeJitConstant("SPATIAL_BINS_X", rp.spatial_bins_x),
        MakeJitConstant("SPATIAL_BINS_Y", rp.spatial_bins_y),
    });

    if (rp.mode == PoolType::DEFORMABLE_BILINEAR)
        jit.AddConstants({MakeJitConstant("TRANS_STD", rp.trans_std),
                          MakeJitConstant("NO_TRANS", rp.no_trans),
                          MakeJitConstant("PART_SIZE", rp.part_size),
                          MakeJitConstant("GROUP_SIZE", rp.group_size)});

    return jit;
}

KernelsData PSROIPoolingKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PSROIPoolingKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
