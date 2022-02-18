// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveMultiSubGraphOpDanglingParams;

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ie_transformation_common_api
 * @brief RemoveMultiSubGraphOpDanglingParams transformation
 * removed MultiSubGraphOp inputs which are not connected to other nodes
 * in the bodies of a MultiSubGraphOp
 */

class ov::pass::RemoveMultiSubGraphOpDanglingParams: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RemoveMultiSubGraphOpDanglingParams();
};
