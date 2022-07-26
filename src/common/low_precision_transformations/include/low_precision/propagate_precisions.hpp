// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <low_precision/lpt_visibility.hpp>
#include "low_precision/rt_info/attribute_parameters.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PropagatePrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PropagatePrecisions transformation propagates PrecisionsAttribute attribute instances precision preserved operations.
 *
 * For more details about the transformation, refer to
 * [PropagatePrecisions](@ref openvino_docs_OV_UG_lpt_PropagatePrecisions) page
 * in the Inference Engine Developer Guide.
 */
class ngraph::pass::low_precision::PropagatePrecisions : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PropagatePrecisions(const AttributeParameters& params = AttributeParameters());
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

private:
    const AttributeParameters params;
};
