// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_state.h"
#include "dnnl_extension_utils.h"
#include "blob_factory.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

void VariableState::Reset() {
    std::memset(state->buffer(), 0, state->byteSize());
}

}   // namespace intel_cpu
}   // namespace ov

