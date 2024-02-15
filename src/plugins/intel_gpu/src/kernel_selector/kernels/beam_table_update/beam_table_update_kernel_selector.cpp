// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "beam_table_update_kernel_selector.hpp"

#include "beam_table_update_kernel_ref.hpp"

namespace kernel_selector {

beam_table_update_kernel_selector::beam_table_update_kernel_selector() {
    Attach<BeamTableUpdateKernelRef>();
}

KernelsData beam_table_update_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::BEAM_TABLE_UPDATE);
}

beam_table_update_kernel_selector& beam_table_update_kernel_selector::Instance() {
    static beam_table_update_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
