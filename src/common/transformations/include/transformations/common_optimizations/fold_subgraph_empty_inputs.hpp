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

class TRANSFORMATIONS_API FoldSubgraphEmptyInputs;

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ie_transformation_common_api
 * @brief FoldSubgraphEmptyInputs transformation fold MultiSubGraphOp inputs (by replacing with Constant op)
 * if the dimension is static and at least one is "0".
 * It means that subgraphs producing empty tensors are removed.
 */

class ov::pass::FoldSubgraphEmptyInputs : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FoldSubgraphEmptyInputs", "0");
    FoldSubgraphEmptyInputs();
};
