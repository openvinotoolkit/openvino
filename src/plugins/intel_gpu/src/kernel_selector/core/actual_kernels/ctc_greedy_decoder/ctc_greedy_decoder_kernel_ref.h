// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ctc_greedy_decoder_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
class CTCGreedyDecoderKernelRef : public CTCGreedyDecoderKernelBase {
public:
    using Parent = CTCGreedyDecoderKernelBase;
    CTCGreedyDecoderKernelRef() : CTCGreedyDecoderKernelBase("ctc_greedy_decoder_ref") {}
    virtual ~CTCGreedyDecoderKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
