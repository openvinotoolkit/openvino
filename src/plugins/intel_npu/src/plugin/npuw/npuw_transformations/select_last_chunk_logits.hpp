// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class SelectLastChunkLogits : public ov::pass::ModelPass {
    uint32_t m_batch_dim;
    std::size_t m_chunk_size;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::SelectLastChunkLogits");
    explicit SelectLastChunkLogits(uint32_t batch_dim, std::size_t chunk_size);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw