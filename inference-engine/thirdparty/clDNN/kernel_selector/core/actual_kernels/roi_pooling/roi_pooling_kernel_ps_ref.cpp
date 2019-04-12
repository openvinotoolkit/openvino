/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "roi_pooling_kernel_ps_ref.h"

namespace kernel_selector {

    ParamsKey PSROIPoolingKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::brfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableDifferentTypes();
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolType(PoolType::BILINEAR);
        k.EnablePositionSensitivePooling();
        return k;
    }

    JitConstants PSROIPoolingKernelRef::GetJitConstants(const roi_pooling_params& rp) const
    {
        JitConstants jit = ROIPoolingKernelBase::GetJitConstants(rp);

        jit.AddConstants({ MakeJitConstant("SPATIAL_BINS_X", rp.spatial_bins_x),
                           MakeJitConstant("SPATIAL_BINS_Y", rp.spatial_bins_y),
                         });

        return jit;
    }

    KernelsData PSROIPoolingKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options, FORCE_PRIORITY_9);
    }
}
