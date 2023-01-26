// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mixed_affinity_utils.hpp"
#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class MixedAffinity: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
    static std::unordered_map<mixed_affinity::Characteristics, mixed_affinity::Subgraph> formSubgraphs(const std::shared_ptr<ov::Model>& m);
};
}  // namespace intel_cpu
}  // namespace ov
