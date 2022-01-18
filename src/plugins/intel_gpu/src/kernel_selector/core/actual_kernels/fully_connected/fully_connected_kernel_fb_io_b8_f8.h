// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_block_kernel_base.h"

namespace kernel_selector {

class FullyConnected_fb_io_b8_f8 : public FullyConnectedBlockKernelBase {
public:
    FullyConnected_fb_io_b8_f8() : FullyConnectedBlockKernelBase("fully_connected_gpu_fb_io_b8_f8_vload") {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const fully_connected_params& arg, int autoTuneIndex = -1) const override;
    size_t GetBatchesPerWorkItem(const fully_connected_params& params) const override;
};
}  // namespace kernel_selector
