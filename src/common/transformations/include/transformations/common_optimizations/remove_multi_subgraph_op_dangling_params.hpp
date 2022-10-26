// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

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

class ov::pass::RemoveMultiSubGraphOpDanglingParams : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveMultiSubGraphOpDanglingParams", "0");
    RemoveMultiSubGraphOpDanglingParams();
};
