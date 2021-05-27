// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory_state.h"
#include "mkldnn_extension_utils.h"
#include "blob_factory.hpp"

using namespace InferenceEngine;

namespace MKLDNNPlugin {

void  MKLDNNVariableState::Reset() {
    std::memset(state->buffer(), 0, state->byteSize());
}

}  // namespace MKLDNNPlugin
