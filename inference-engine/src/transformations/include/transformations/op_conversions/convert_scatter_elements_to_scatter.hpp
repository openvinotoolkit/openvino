// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertScatterElementsToScatter;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertScatterElementsToScatter convert opset3::ScatterElementsUpdate to opset3::ScatterUpdate.
 */
class ov::pass::ConvertScatterElementsToScatter: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertScatterElementsToScatter();
};
