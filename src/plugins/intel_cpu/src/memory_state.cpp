// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_state.h"

#include "dnnl_extension_utils.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

void VariableState::reset() {
    // state = make_blob_with_precision(tensor_desc);
    // state->allocate();
    // std::memset(state->buffer(), 0, state->byteSize());
}

}  // namespace intel_cpu
}  // namespace ov
