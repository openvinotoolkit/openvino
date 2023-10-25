// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertNMS5ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      Convert NMS-5 directly to inner NMS.
 */

class ngraph::pass::ConvertNMS5ToLegacyMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS5ToLegacyMatcher", "0");
    ConvertNMS5ToLegacyMatcher(bool force_i32_output_type = true);
};
