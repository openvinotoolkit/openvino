// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API LabelResolvingThroughSelect;
class TRANSFORMATIONS_API SymbolicOptimizations;
class TRANSFORMATIONS_API SymbolicPropagation;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Runs optimizations which are based on symbolic shape inference
 */
class ov::pass::SymbolicOptimizations : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SymbolicOptimizations", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Assigns labels / symbols to all tensors on shapes and values. Uses shape inference and other special rules to
 * propagate labels / symbols
 */
class ov::pass::SymbolicPropagation : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SymbolicPropagation");
    SymbolicPropagation();
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::shared_ptr<ov::TableOfEquivalence> m_te;
};

class ov::pass::LabelResolvingThroughSelect : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LabelResolvingThroughSelect", "0");
    LabelResolvingThroughSelect();
};