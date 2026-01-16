// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

/**
 * @brief MoE I/O tensors structure
 *
 * Contains input and output tensors for MoE expert inference.
 * Shared between IBaseInferRequest and MoEExecutor.
 */
struct MoEIO {
    std::vector<ov::SoPtr<ov::ITensor>> outputs;  // # of elements - # of subgraph outputs
    ov::SoPtr<ov::ITensor> router_scores;         // Expert model input: router output for expert selection
    ov::SoPtr<ov::ITensor> expert_input;          // Expert model input: token embeddings
};

}  // namespace npuw
}  // namespace ov
