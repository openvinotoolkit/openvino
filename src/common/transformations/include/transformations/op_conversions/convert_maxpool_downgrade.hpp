// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMaxPool8ToMaxPool1;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertMaxPool8ToMaxPool1 converts v8::MaxPool into v1::MaxPool.
 */
class ngraph::pass::ConvertMaxPool8ToMaxPool1 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMaxPool8ToMaxPool1");
    ConvertMaxPool8ToMaxPool1();
};
