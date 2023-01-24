// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/graph_rewrite.hpp>
#include "mixed_affinity_subgraph.hpp"

namespace ov {
namespace intel_cpu {
class SwitchAffinity: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwitchAffinity(const std::unordered_map<mixed_affinity::Characteristics, mixed_affinity::Subgraph>& subgraphs, const bool share_constants = true);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool share_constants;
    std::unordered_map<mixed_affinity::Characteristics, mixed_affinity::Subgraph> subgraphs;
};
}  // namespace intel_cpu
}  // namespace ov
