// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ApplyTableOfEquivalence;
class TRANSFORMATIONS_API OptimizeLabelsUsedAsValues;
}  // namespace pass
}  // namespace ov

class ov::pass::ApplyTableOfEquivalence : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ApplyTableOfEquivalence", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ov::pass::OptimizeLabelsUsedAsValues : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("OptimizeLabelsUsedAsValues", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};