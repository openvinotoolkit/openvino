// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMaxPool1ToMaxPool8;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertMaxPool1ToMaxPool8 converts v1::MaxPool into v8::MaxPool.
 */

class ngraph::pass::ConvertMaxPool1ToMaxPool8 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMaxPool1ToMaxPool8");
    ConvertMaxPool1ToMaxPool8();
};
