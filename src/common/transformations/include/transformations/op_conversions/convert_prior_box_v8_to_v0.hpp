// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <openvino/core/ov_visibility.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API ConvertPriorBox8To0;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertPriorBox8To1 converts v8::PriorBox into v0::PriorBox.
 */
class ngraph::pass::ConvertPriorBox8To0 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBox8To0();
};
