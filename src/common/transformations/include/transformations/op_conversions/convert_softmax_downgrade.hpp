// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

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
    OPENVINO_RTTI("ConvertSoftMax8ToSoftMax1", "0");
    ConvertSoftMax8ToSoftMax1();
};
