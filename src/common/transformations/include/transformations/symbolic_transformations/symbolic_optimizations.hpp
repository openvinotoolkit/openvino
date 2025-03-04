// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API SymbolicOptimizations;
class TRANSFORMATIONS_API SymbolicPropagation;
class TRANSFORMATIONS_API LabelResolvingThroughSelect;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Runs optimizations which are based on symbolic shape inference
 */
class ov::pass::SymbolicOptimizations : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SymbolicOptimizations");
    explicit SymbolicOptimizations(bool full_run = true);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
    std::shared_ptr<ov::pass::Manager> get_manager() {
        return m_manager;
    };

private:
    std::shared_ptr<ov::pass::Manager> m_manager;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Assigns labels / symbols to all tensors on shapes and values. Uses shape inference and other special rules to
 * propagate labels / symbols
 */
class ov::pass::SymbolicPropagation : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SymbolicPropagation");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Transformation requires equal labels on one input of Add and output of last Reshape in the pattern:
 *      -> Add -> Reshape -[then or else input]-> Select -> Softmax -> Reshape ->
 *
 * If shape labels onn mentioned tensors are equal we proved that no broadcasting of this input was done for Add and
 * for Select. Therefore, we can put the same labels on the output of Add and Select. This transformation helps
 * propagate labels and will not be needed if we would use information on equality of products of input and output
 * dimensions of Reshape operations
 */
class ov::pass::LabelResolvingThroughSelect : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("LabelResolvingThroughSelect");
    LabelResolvingThroughSelect();
};
