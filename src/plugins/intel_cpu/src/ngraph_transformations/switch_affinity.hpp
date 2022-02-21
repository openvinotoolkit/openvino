// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
struct Subgraph {
    std::set<ov::Input<ov::Node>> starts;
    std::set<ov::Output<ov::Node>> ends;
};

class SwitchAffinity: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwitchAffinity(const std::unordered_map<size_t, Subgraph>& subgraphs, const bool share_constants = true);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool share_constants;
    std::unordered_map<size_t, Subgraph> subgraphs;
};
}  // namespace intel_cpu
}  // namespace ov
