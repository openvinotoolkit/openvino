// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertSoftMax1ToSoftMax8;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertSoftMax1ToSoftMax8 converts v1::SoftMax into v8::SoftMax.
 */

class ngraph::pass::ConvertSoftMax1ToSoftMax8 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSoftMax1ToSoftMax8", "0");
    ConvertSoftMax1ToSoftMax8();
};
