// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertDeformableConv8To1;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertDeformableConv8To1 converts v8::DeformableConvolution into v1::DeformableConvolution.
 */
class ov::pass::ConvertDeformableConv8To1 : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDeformableConv8To1();
};
