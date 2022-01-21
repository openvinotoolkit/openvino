// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertSoftMax8ToSoftMax1;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertSoftMax8ToSoftMax1 converts v8::SoftMax into v1::SoftMax.
 */
class ngraph::pass::ConvertSoftMax8ToSoftMax1 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertSoftMax8ToSoftMax1();
};
