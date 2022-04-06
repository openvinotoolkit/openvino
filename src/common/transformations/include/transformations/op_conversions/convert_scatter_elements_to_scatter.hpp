// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertScatterElementsToScatter;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertScatterElementsToScatter convert opset3::ScatterElementsUpdate to opset3::ScatterUpdate.
 */
class ngraph::pass::ConvertScatterElementsToScatter : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertScatterElementsToScatter", "0");
    ConvertScatterElementsToScatter();
};
