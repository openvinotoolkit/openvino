// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation replaces KV-cache processing part in LLMs by PagedAttention operation.
 * NOTE:
 * The transformation may throw an exception when some configuration of the model failed:
 * i.e. the SDPA node is absent in the model. This means the graph cannot be processed for the PA scenario,
 * so the GenAI pipeline (the only pipeline the transformation is used in so far) will fallback to the SDPA
 * implementaion and run inference using it.
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API SDPAToPagedAttention : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAToPagedAttention");

    explicit SDPAToPagedAttention(bool use_per_layer_block_indices_inputs = false,
                                  bool use_score_outputs = false,
                                  bool allow_score_aggregation = false,
                                  bool allow_cache_rotation = false,
                                  bool allow_xattention = false,
                                  bool allow_adaptive_rkv = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_use_per_layer_block_indices_inputs;
    bool m_use_score_outputs;
    bool m_allow_score_aggregation;
    bool m_allow_cache_rotation;
    bool m_allow_xattention;
    bool m_allow_adaptive_rkv;
};
}  // namespace pass
}  // namespace ov
