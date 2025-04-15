// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ApplySymbolEquivalence;
class TRANSFORMATIONS_API OptimizeSymbolsUsedAsValues;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Resets symbols on output shapes and values according to symbol equivalence. It
 * allows to reduce number of labels used in the model and to disambiguate symbol values.
 */
class ov::pass::ApplySymbolEquivalence : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ApplySymbolEquivalence");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Collects sources where each symbol initially appeared (on shape or shape sub-graph) and attaches all
 * value usages of this label to this initial source
 */
class ov::pass::OptimizeSymbolsUsedAsValues : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("OptimizeSymbolsUsedAsValues");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
