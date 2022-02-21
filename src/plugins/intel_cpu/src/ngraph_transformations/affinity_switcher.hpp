// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class AffinitySwitcher : public ov::pass::ModelPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AffinitySwitcher(const bool share_constants = true);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool share_constants;
};
}  // namespace intel_cpu
}  // namespace ov
