// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_block_kernel_base.h"

namespace kernel_selector {

class FullyConnected_bs_f_bsv16_af8 : public FullyConnectedBlockKernelBase {
public:
    FullyConnected_bs_f_bsv16_af8() : FullyConnectedBlockKernelBase("fully_connected_gpu_bs_f_bsv16_af8_vload") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const fully_connected_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector