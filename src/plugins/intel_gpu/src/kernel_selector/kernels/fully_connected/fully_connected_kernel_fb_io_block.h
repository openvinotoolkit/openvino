// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

class FullyConnected_fb_io_block : public FullyConnectedKernelBase {
public:
    FullyConnected_fb_io_block() : FullyConnectedKernelBase("fully_connected_gpu_fb_io_block_fp16") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const fully_connected_params& params,
                                 const FullyConnectedKernelBase::DispatchData& dispatchData) const override;
    DispatchData SetDefault(const fully_connected_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
