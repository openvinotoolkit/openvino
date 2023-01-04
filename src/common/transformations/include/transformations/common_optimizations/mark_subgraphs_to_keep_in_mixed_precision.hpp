// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkSugraphsToKeepInMixedPrecision;

// through this nodes precision sensitiveness is propagated
extern std::shared_ptr<Node> propagate_through_ops;

}  // namespace pass
}  // namespace ov

/* MarkSugraphsToKeepInMixedPrecision  marks patterns that should be kept in f32 for
 * mixed precision inference. Such patterns include: L2Normalize, MVN, Division with small eps values
 */
class ov::pass::MarkSugraphsToKeepInMixedPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkSugraphsToKeepInMixedPrecision", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
