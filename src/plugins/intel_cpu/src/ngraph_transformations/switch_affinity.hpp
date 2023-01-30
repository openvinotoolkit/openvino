// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/graph_rewrite.hpp>
#include "mixed_affinity_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
class SwitchAffinity: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwitchAffinity(const std::unordered_map<Properties, Subgraph>& subgraphs, const bool share_constants = true);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool share_constants;
    std::unordered_map<Properties, Subgraph> subgraphs;
};
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
