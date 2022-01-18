// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_greedy_decoder_kernel_selector.h"
#include "ctc_greedy_decoder_kernel_ref.h"

namespace kernel_selector {
ctc_greedy_decoder_kernel_selector::ctc_greedy_decoder_kernel_selector() {
    Attach<CTCGreedyDecoderKernelRef>();
}

KernelsData ctc_greedy_decoder_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::CTC_GREEDY_DECODER);
}
}  // namespace kernel_selector
