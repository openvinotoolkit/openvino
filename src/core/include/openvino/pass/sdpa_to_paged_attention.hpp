// Copyright (C) 2018-2024 Intel Corporation
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
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API SDPAToPagedAttention : public ModelPass {
public:
    OPENVINO_RTTI("SDPAToPagedAttention");

    SDPAToPagedAttention(bool use_block_indices_inputs = false, bool use_score_outputs = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_use_block_indices_inputs;
    bool m_use_score_outputs;
};
}  // namespace pass
}  // namespace ov
