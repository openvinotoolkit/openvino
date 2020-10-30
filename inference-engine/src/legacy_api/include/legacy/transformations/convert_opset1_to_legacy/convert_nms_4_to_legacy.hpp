// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class INFERENCE_ENGINE_API_CLASS(ConvertNMS4ToLegacyMatcher);

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      Convert NMS-4 directly to legacy NMS because NMS-3 and NMS-1 have different shape infer function
 */


class ngraph::pass::ConvertNMS4ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMS4ToLegacyMatcher();
};

