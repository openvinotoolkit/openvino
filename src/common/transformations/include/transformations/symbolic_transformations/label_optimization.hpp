// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ApplyTableOfEquivalence;
class TRANSFORMATIONS_API OptimizeLabelsUsedAsValues;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Resets symbols / labels on output shapes and values according to table of symbol / label equivalence. It
 * allows to reduce number of labels used in the model and to disambiguate label values.
 */
class ov::pass::ApplyTableOfEquivalence : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ApplyTableOfEquivalence", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Collects sources where each symbol / label initially appeared (on shape or shape sub-graph) and attaches all
 * value usages of this label to this initial source
 */
class ov::pass::OptimizeLabelsUsedAsValues : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("OptimizeLabelsUsedAsValues", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};