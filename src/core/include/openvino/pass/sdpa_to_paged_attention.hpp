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

    SDPAToPagedAttention(bool use_cache_eviction = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_use_cache_eviction;
};
}  // namespace pass
}  // namespace ov
