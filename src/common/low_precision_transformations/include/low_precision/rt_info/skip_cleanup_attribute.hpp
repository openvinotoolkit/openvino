// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ngraph {
/**
 * @ingroup ie_transformation_common_api
 * @brief SkipCleanupAttribute defines if operation <put here>.
 *
 * For more details about the attribute, refer to
 * [SkipCleanupAttribute](@ref openvino_docs_IE_DG_lpt_SkipCleanupAttribute) page in the Inference Engine Developer
 * Guide.
 */
class LP_TRANSFORMATIONS_API SkipCleanupAttribute : public ov::RuntimeAttribute {
    bool skip;

public:
    OPENVINO_RTTI("LowPrecision::SkipCleanup", "", ov::RuntimeAttribute, 0);
    SkipCleanupAttribute(const bool skip);

    static ov::Any create(const std::shared_ptr<ngraph::Node>& node, const bool skip);
    // vizualize shared attributes details in VizualizeTree pass
    std::string to_string() const override;
    const bool value() const;
};
} // namespace ngraph
