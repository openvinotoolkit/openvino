// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class IncreasePrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePrecision");
    IncreasePrecision();
protected:
    bool insert_converts_before_if_needed(const std::shared_ptr<ov::Node>& node,
                const ov::element::Type desired_et,
                size_t& input_idx,
                const std::vector<size_t>& skip_inputs = {});
    void insert_converts_after_if_needed(const std::shared_ptr<ov::Node>& node, const ov::element::Type original_et, size_t& output_idx);
};
}   // namespace ov::intel_gpu