// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class PrepareQSAIndexCache : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareQSAIndexCache");
    explicit PrepareQSAIndexCache(size_t context_size) : m_context_size(context_size) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    size_t m_context_size;
};

}  // namespace ov::npuw
