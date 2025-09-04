// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertTypeRelaxedToBase transformation converts type_relaxed operations
 * to their base opset equivalents before serialization.
 * 
 * This transformation is used to prevent the "Cannot create IsInf layer from 
 * unsupported opset: type_relaxed_opset" error during model import/export.
 * 
 * Type relaxed operations are internal operations that shouldn't be serialized
 * to IR format. This pass converts them to their base opset versions for
 * compatibility.
 */
class TRANSFORMATIONS_API ConvertTypeRelaxedToBase : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTypeRelaxedToBase", "0");
    ConvertTypeRelaxedToBase();
};

}  // namespace pass
}  // namespace ov
