// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMVN1ToMVN6;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertMVN1ToMVN6 covert v0:MVN into v6::MVN.
 */
class ov::pass::ConvertMVN1ToMVN6 : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMVN1ToMVN6();
};
