// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CommonOptimizations;

}  // namespace pass
}  // namespace ov

class ov::pass::CommonOptimizations : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("CommonOptimizations", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};

namespace ngraph {
namespace pass {
using ov::pass::CommonOptimizations;
}  // namespace pass
}  // namespace ngraph
