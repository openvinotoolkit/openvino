// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FoldSubgraphEmptyInputs;
class TRANSFORMATIONS_API DisableFoldSubgraphEmptyInputs;

TRANSFORMATIONS_API void disable_fold_subgraph_empty_inputs(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_fold_subgraph_empty_inputs(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool fold_subgraph_empty_inputs_is_disabled(const std::shared_ptr<Node>& node);

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ov_transformation_common_api
 * @brief FoldSubgraphEmptyInputs transformation fold MultiSubGraphOp inputs (by replacing with Constant op)
 * if the dimension is static and at least one is "0".
 * It means that subgraphs producing empty tensors are removed.
 */

class ov::pass::FoldSubgraphEmptyInputs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FoldSubgraphEmptyInputs");
    FoldSubgraphEmptyInputs();
};

class ov::pass::DisableFoldSubgraphEmptyInputs : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("DisableFoldSubgraphEmptyInputs", "0", ov::RuntimeAttribute);
    DisableFoldSubgraphEmptyInputs() = default;
    bool is_copyable() const override {
        return false;
    }
};
