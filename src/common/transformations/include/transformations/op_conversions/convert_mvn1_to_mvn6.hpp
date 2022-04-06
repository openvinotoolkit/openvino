// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMVN1ToMVN6;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertMVN1ToMVN6 covert v0:MVN into v6::MVN.
 */
class ngraph::pass::ConvertMVN1ToMVN6 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMVN1ToMVN6", "0");
    ConvertMVN1ToMVN6();
};
