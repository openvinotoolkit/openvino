// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node_output.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class QwenSparseAttentionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::QwenSparseAttentionFusion");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
    void lower_indexers();
    size_t fused_count() const {
        return m_indexers.size();
    }

private:
    struct Indexer {
        std::shared_ptr<ov::op::v17::SparseAttentionIndexer> node;
        ov::Output<ov::Node> indices;
        ov::Output<ov::Node> counts;
    };
    std::vector<Indexer> m_indexers;
};

}  // namespace ov::npuw
