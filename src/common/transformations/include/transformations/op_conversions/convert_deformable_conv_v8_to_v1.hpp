// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertDeformableConv8To1;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertDeformableConv8To1 converts v8::DeformableConvolution into v1::DeformableConvolution.
 */
class ngraph::pass::ConvertDeformableConv8To1 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDeformableConv8To1();
};
