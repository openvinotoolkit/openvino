// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_state.h"

#include "dnnl_extension_utils.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

void VariableState::reset() {
    std::memset(m_state->data(), 0, m_state->get_byte_size());
}

}  // namespace intel_cpu
}  // namespace ov
