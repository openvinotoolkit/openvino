// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mixed_affinity_utils.hpp"
#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
class MixedAffinity: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
    static std::unordered_map<Properties, Subgraph> formSubgraphs(const std::shared_ptr<ov::Model>& m);
};
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
