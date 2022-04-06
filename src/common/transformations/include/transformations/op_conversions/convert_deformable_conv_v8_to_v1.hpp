// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

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
    OPENVINO_RTTI("ConvertDeformableConv8To1", "0");
    ConvertDeformableConv8To1();
};
