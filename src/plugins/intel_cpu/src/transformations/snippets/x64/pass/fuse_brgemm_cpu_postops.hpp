// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

class FuseScaleShift : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseScaleShift");
    FuseScaleShift();
};

class FuseBinaryEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseBinaryEltwise");
    FuseBinaryEltwise(std::set<std::shared_ptr<ov::op::v0::Parameter>>& external_params);

private:
    size_t m_fused_postops_count = 0;
    std::set<std::shared_ptr<ov::op::v0::Parameter>>& m_external_params;
};

class FuseBrgemmCPUPostops : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseBrgemmCPUPostops");
    FuseBrgemmCPUPostops(std::set<size_t>& brgemm_external_params_idces)
        : m_brgemm_external_params_idces(brgemm_external_params_idces) {
        add_matcher<FuseScaleShift>();
        add_matcher<FuseBinaryEltwise>(m_external_params);
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::set<size_t>& m_brgemm_external_params_idces;
    // Note: this set is needed to collect external params.
    // This set will be converted to m_external_params_indices at run_on_model stage
    std::set<std::shared_ptr<ov::op::v0::Parameter>> m_external_params = {};
};

}  // namespace ov::intel_cpu::pass
