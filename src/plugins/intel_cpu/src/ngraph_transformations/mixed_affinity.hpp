// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <set>

namespace ov {
namespace intel_cpu {
class MixedAffinity: public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
}  // namespace intel_cpu
}  // namespace ov
